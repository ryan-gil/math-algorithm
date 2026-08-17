import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from math_sft_common import (
    MODEL_ID,
    extract_integer,
    prompt_messages,
    read_math_csv,
)


DEFAULT_VALID = Path(
    "deep-learning-challenge-2026/deep_chal_math_valid_semantic_10.csv"
)
FIELDS = ["id", "prediction", "raw_response", "answer", "correct"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact-match evaluation for a Qwen2.5-3B QLoRA adapter."
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_completed_ids(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected output columns: {reader.fieldnames}")
        return {row["id"] for row in reader}


def load_model(adapter_path):
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU in Colab.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer_source = (
        adapter_path
        if (adapter_path / "tokenizer_config.json").exists()
        else MODEL_ID
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, rows, max_new_tokens):
    prompts = [
        tokenizer.apply_chat_template(
            prompt_messages(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def summarize(path, selected_ids):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["id"] in selected_ids]
    valid = sum(bool(row["prediction"]) for row in rows)
    correct = sum(row["correct"].casefold() == "true" for row in rows)
    accuracy = correct / len(rows) if rows else 0.0
    summary = {
        "rows": len(rows),
        "valid_predictions": valid,
        "correct": correct,
        "exact_match_accuracy": accuracy,
    }
    print(json.dumps(summary, indent=2))
    summary_path = path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    if args.limit < 0:
        raise ValueError("--limit must be 0 or positive")
    if not args.adapter_path.exists():
        raise FileNotFoundError(args.adapter_path)

    rows = read_math_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]
    selected_ids = {row["id"] for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}")
    completed_ids = load_completed_ids(args.output) if args.resume else set()
    pending_rows = [row for row in rows if row["id"] not in completed_ids]
    if not pending_rows:
        summarize(args.output, selected_ids)
        return

    print(f"Model: {MODEL_ID} (fixed)")
    print(f"Adapter: {args.adapter_path}")
    print(f"Pending rows: {len(pending_rows)}/{len(rows)}")
    model, tokenizer = load_model(args.adapter_path)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for start in range(0, len(pending_rows), args.batch_size):
            batch = pending_rows[start : start + args.batch_size]
            responses = generate(model, tokenizer, batch, args.max_new_tokens)
            for row, response in zip(batch, responses):
                prediction = extract_integer(response)
                writer.writerow(
                    {
                        "id": row["id"],
                        "prediction": prediction,
                        "raw_response": response.strip(),
                        "answer": row["answer"],
                        "correct": prediction == row["answer"],
                    }
                )
            file.flush()
            completed = min(start + len(batch), len(pending_rows))
            print(f"Progress: {completed}/{len(pending_rows)}")
    summarize(args.output, selected_ids)


if __name__ == "__main__":
    main()
