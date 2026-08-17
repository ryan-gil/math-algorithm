import argparse
import json
import math
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

from math_sft_common import MODEL_ID, read_math_csv, training_example


DEFAULT_TRAIN = Path(
    "deep-learning-challenge-2026/deep_chal_math_train_semantic_90.csv"
)
DEFAULT_VALID = Path(
    "deep-learning-challenge-2026/deep_chal_math_valid_semantic_10.csv"
)
PLANNED_EPOCHS = 2


class StopAtTargetEpochCallback(TrainerCallback):
    """Keep a two-epoch scheduler while allowing inspection after epoch one."""

    def __init__(self, target_epoch):
        self.target_epoch = target_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.target_epoch - 1e-6:
            control.should_save = True
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-3B-Instruct math answers."
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Trainer checkpoint path, or 'latest' inside --output-dir.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU. In Colab select "
            "Runtime > Change runtime type > T4 GPU."
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute dtype: {dtype}")
    return dtype


def limit_rows(rows, limit, name):
    if limit < 0:
        raise ValueError(f"--max-{name}-samples must be 0 or positive")
    return rows if limit == 0 else rows[:limit]


def ensure_disjoint(train_rows, valid_rows):
    overlap = {row["id"] for row in train_rows} & {row["id"] for row in valid_rows}
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"Train/valid IDs overlap, for example: {examples}")


def percentile(sorted_values, percent):
    if not sorted_values:
        return 0
    index = math.ceil((percent / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, index)]


def report_token_lengths(tokenizer, rows, max_length, name):
    lengths = []
    for row in rows:
        example = training_example(row)
        messages = example["prompt"] + example["completion"]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))
    lengths.sort()
    truncated = sum(length > max_length for length in lengths)
    print(
        f"{name} tokens: p50={percentile(lengths, 50)}, "
        f"p95={percentile(lengths, 95)}, p99={percentile(lengths, 99)}, "
        f"max={lengths[-1]}, over max_length={truncated}/{len(lengths)}"
    )
    return {
        "p50": percentile(lengths, 50),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": lengths[-1],
        "over_max_length": truncated,
    }


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
        raise FileNotFoundError(f"No complete Trainer checkpoint found in {output_dir}")
    return max(checkpoints, key=checkpoint_step)


def resolve_resume_checkpoint(args):
    if not args.resume_from_checkpoint:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use --resume-from-checkpoint latest or choose a new directory."
            )
        return None

    checkpoint = (
        latest_checkpoint(args.output_dir)
        if args.resume_from_checkpoint.casefold() == "latest"
        else Path(args.resume_from_checkpoint)
    )
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Not a resumable Trainer checkpoint: {checkpoint}")
    with state_path.open("r", encoding="utf-8") as file:
        completed_epoch = float(json.load(file).get("epoch") or 0)
    if completed_epoch >= args.target_epochs - 1e-6:
        raise ValueError(
            f"Checkpoint already reached epoch {completed_epoch:g}; "
            f"target is {args.target_epochs}."
        )
    print(f"Resuming from: {checkpoint} (completed epoch {completed_epoch:g})")
    return checkpoint


def main():
    args = parse_args()
    dtype = require_cuda()
    resume_checkpoint = resolve_resume_checkpoint(args)

    train_rows = limit_rows(
        read_math_csv(args.train_file), args.max_train_samples, "train"
    )
    valid_rows = limit_rows(
        read_math_csv(args.valid_file), args.max_eval_samples, "eval"
    )
    ensure_disjoint(train_rows, valid_rows)
    print(f"Model: {MODEL_ID} (fixed)")
    print(f"Train rows: {len(train_rows)} | valid rows: {len(valid_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_stats = {
        "train": report_token_lengths(tokenizer, train_rows, args.max_length, "Train"),
        "valid": report_token_lengths(tokenizer, valid_rows, args.max_length, "Valid"),
    }

    train_dataset = Dataset.from_list([training_example(row) for row in train_rows])
    valid_dataset = Dataset.from_list([training_example(row) for row in valid_rows])

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
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
        # Both stages share this schedule. Stage one stops through the callback.
        num_train_epochs=PLANNED_EPOCHS,
        max_length=args.max_length,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        packing=False,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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
        eval_strategy="epoch",
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
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[StopAtTargetEpochCallback(args.target_epochs)],
    )
    trainer.model.print_trainable_parameters()
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    eval_metrics = trainer.evaluate()

    adapter_dir = args.output_dir / f"adapter-epoch-{args.target_epochs}"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    final_checkpoint = latest_checkpoint(args.output_dir)
    summary = {
        "model": MODEL_ID,
        "target_epochs": args.target_epochs,
        "planned_epochs": PLANNED_EPOCHS,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "max_length": args.max_length,
        "token_stats": token_stats,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
        "latest_trainer_checkpoint": str(final_checkpoint),
        "adapter_dir": str(adapter_dir),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path = args.output_dir / f"run-summary-epoch-{args.target_epochs}.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Resume checkpoint: {final_checkpoint}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
