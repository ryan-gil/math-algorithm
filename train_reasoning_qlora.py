import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import (
    PROJECT_DIR,
    STUDENT_MODEL_ID,
    extract_final_answer,
    student_prompt_messages,
)


DEFAULT_TRAIN = PROJECT_DIR / "outputs(2)" / "reasoning" / "reasoning_train_sft.csv"
PLANNED_EPOCHS = 2


class StopAtTargetEpochCallback(TrainerCallback):
    def __init__(self, target_epoch):
        self.target_epoch = target_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.target_epoch - 1e-6:
            control.should_save = True
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reasoning QLoRA SFT for Qwen2.5-3B-Instruct."
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, choices=(1, 2), required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--allow-truncation", action="store_true")
    return parser.parse_args()


def read_sft_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"id", "question", "answer", "completion", "solution_source"}
    if not rows:
        raise ValueError(f"No SFT rows found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing SFT columns: {sorted(missing)}")

    seen = set()
    for line_number, row in enumerate(rows, start=2):
        row_id = row["id"].strip()
        question = row["question"].strip()
        completion = row["completion"].strip()
        answer = row["answer"].strip()
        parsed, method = extract_final_answer(completion, allow_fallback=False)
        if not row_id or not question or not completion:
            raise ValueError(f"Empty SFT value at CSV line {line_number}")
        if row_id in seen:
            raise ValueError(f"Duplicate SFT ID: {row_id}")
        if method != "final_answer" or parsed != answer:
            raise ValueError(f"Completion/answer mismatch for {row_id}")
        seen.add(row_id)
    return rows


def training_example(row):
    return {
        "prompt": student_prompt_messages(row["question"]),
        "completion": [{"role": "assistant", "content": row["completion"]}],
    }


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute dtype: {dtype}")
    return dtype


def percentile(values, percent):
    index = math.ceil((percent / 100) * len(values)) - 1
    return values[max(0, index)]


def report_token_lengths(tokenizer, rows, max_length):
    lengths = []
    for row in rows:
        example = training_example(row)
        token_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))
    lengths.sort()
    stats = {
        "p50": percentile(lengths, 50),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": lengths[-1],
        "over_max_length": sum(length > max_length for length in lengths),
    }
    print("Token lengths: " + json.dumps(stats))
    return stats


def checkpoint_step(path):
    try:
        return int(path.name.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(output_dir):
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No complete checkpoint in {output_dir}")
    return max(checkpoints, key=checkpoint_step)


def resolve_resume(args):
    if not args.resume_from_checkpoint:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}; resume or use a new directory"
            )
        return None
    checkpoint = (
        latest_checkpoint(args.output_dir)
        if args.resume_from_checkpoint.casefold() == "latest"
        else Path(args.resume_from_checkpoint)
    )
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Not a resumable checkpoint: {checkpoint}")
    completed_epoch = float(json.loads(state_path.read_text())["epoch"] or 0)
    if completed_epoch >= args.target_epochs - 1e-6:
        raise ValueError(
            f"Checkpoint already reached epoch {completed_epoch:g}; target is {args.target_epochs}"
        )
    print(f"Resuming from {checkpoint} at epoch {completed_epoch:g}")
    return checkpoint


def main():
    args = parse_args()
    if args.max_train_samples < 0:
        raise ValueError("--max-train-samples must be zero or positive")
    resume_checkpoint = resolve_resume(args)
    rows = read_sft_csv(args.train_file)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    print(f"Model: {STUDENT_MODEL_ID} (fixed)")
    print(f"Training rows: {len(rows)}")

    dtype = require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_stats = report_token_lengths(tokenizer, rows, args.max_length)
    if token_stats["over_max_length"] and not args.allow_truncation:
        raise ValueError(
            f"{token_stats['over_max_length']} training rows exceed --max-length "
            f"{args.max_length}. Increase --max-length or rebuild with shorter solutions. "
            "Use --allow-truncation only after reviewing the affected data."
        )
    dataset = Dataset.from_list([training_example(row) for row in rows])

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=PLANNED_EPOCHS,
        max_length=args.max_length,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        packing=False,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[StopAtTargetEpochCallback(args.target_epochs)],
    )
    trainer.model.print_trainable_parameters()
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )

    adapter_dir = args.output_dir / f"adapter-epoch-{args.target_epochs}"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    final_checkpoint = latest_checkpoint(args.output_dir)
    summary = {
        "model": STUDENT_MODEL_ID,
        "target_epochs": args.target_epochs,
        "train_rows": len(rows),
        "train_metrics": train_result.metrics,
        "token_stats": token_stats,
        "adapter_dir": str(adapter_dir),
        "final_checkpoint": str(final_checkpoint),
    }
    summary_path = args.output_dir / f"run-summary-epoch-{args.target_epochs}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
