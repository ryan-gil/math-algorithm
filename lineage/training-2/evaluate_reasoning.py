import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import (
    DEFAULT_VALID,
    STUDENT_MODEL_ID,
    ensure_parent,
    extract_final_answer,
    read_math_csv,
    student_prompt_messages,
)


FIELDS = [
    "id",
    "prediction",
    "parser_method",
    "raw_response",
    "answer",
    "correct",
    "truncated",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact-match evaluation for a Qwen2.5-3B reasoning adapter."
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_completed(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected output columns: {reader.fieldnames}")
        return {row["id"] for row in reader}


def load_model(adapter_path):
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else STUDENT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, rows, args):
    prompts = [
        tokenizer.apply_chat_template(
            student_prompt_messages(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [
        (response.strip(), not bool((token_ids == eos_id).any().item()))
        for response, token_ids in zip(responses, generated)
    ]


def summarize(path, selected_ids):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["id"] in selected_ids]
    correct = sum(row["correct"].casefold() == "true" for row in rows)
    parse_failures = sum(not row["prediction"] for row in rows)
    truncated = sum(row["truncated"].casefold() == "true" for row in rows)
    marker_parses = sum(row["parser_method"] == "final_answer" for row in rows)
    summary = {
        "rows": len(rows),
        "correct": correct,
        "exact_match_accuracy": correct / len(rows) if rows else 0.0,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "final_marker_parses": marker_parses,
        "truncated": truncated,
        "truncation_rate": truncated / len(rows) if rows else 0.0,
    }
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    if not args.adapter_path.exists():
        raise FileNotFoundError(args.adapter_path)

    rows = read_math_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]
    selected_ids = {row["id"] for row in rows}
    ensure_parent(args.output)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or --overwrite")
    completed = load_completed(args.output) if args.resume else set()
    pending = [row for row in rows if row["id"] not in completed]
    if not pending:
        summarize(args.output, selected_ids)
        return

    model, tokenizer = load_model(args.adapter_path)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            generated = generate(model, tokenizer, batch, args)
            for row, (response, truncated) in zip(batch, generated):
                prediction, method = extract_final_answer(response, allow_fallback=True)
                writer.writerow(
                    {
                        "id": row["id"],
                        "prediction": prediction,
                        "parser_method": method,
                        "raw_response": response,
                        "answer": row["answer"],
                        "correct": prediction == row["answer"],
                        "truncated": truncated,
                    }
                )
            file.flush()
            print(f"Progress: {min(start + len(batch), len(pending))}/{len(pending)}")
    summarize(args.output, selected_ids)


if __name__ == "__main__":
    main()
