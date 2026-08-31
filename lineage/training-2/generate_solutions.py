import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import (
    DEFAULT_TRAIN,
    PROJECT_DIR,
    TEACHER_MODEL_ID,
    ensure_parent,
    extract_final_answer,
    read_math_csv,
    teacher_messages,
)


DEFAULT_OUTPUT = PROJECT_DIR / "outputs(2)" / "reasoning" / "solution_candidates.csv"
FIELDS = [
    "id",
    "candidate_index",
    "question",
    "answer",
    "solution",
    "parsed_answer",
    "parser_method",
    "answer_match",
    "finish_status",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate resumable math solution candidates with Qwen2.5-7B-Instruct."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=TEACHER_MODEL_ID)
    parser.add_argument("--candidates-per-question", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.candidates_per_question < 1:
        raise ValueError("--candidates-per-question must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_input_tokens < 128 or args.max_new_tokens < 16:
        raise ValueError("Token limits are too small")
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")


def load_completed(path, source_by_id):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected output columns in {path}: {reader.fieldnames}")
        completed = set()
        for row in reader:
            source = source_by_id.get(row["id"])
            if source is None:
                raise ValueError(f"Output contains an ID not present in the input: {row['id']}")
            if row["question"] != source["question"] or row["answer"] != source["answer"]:
                raise ValueError(f"Output/input mismatch for {row['id']}; do not resume")
            key = (row["id"], int(row["candidate_index"]))
            if key in completed:
                raise ValueError(f"Duplicate candidate key in output: {key}")
            completed.add(key)
        return completed


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit Qwen2.5-7B generation requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute dtype: {dtype}")
    return dtype


def load_model(model_id, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, rows, args):
    prompts = [
        tokenizer.apply_chat_template(
            teacher_messages(row["question"]),
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
    generation_kwargs = {
        "do_sample": args.candidates_per_question > 1,
        "num_return_sequences": args.candidates_per_question,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    if args.candidates_per_question > 1:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_width:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    records = []
    for row_index, row in enumerate(rows):
        for candidate_index in range(args.candidates_per_question):
            output_index = row_index * args.candidates_per_question + candidate_index
            token_ids = generated[output_index]
            solution = texts[output_index].strip()
            parsed, method = extract_final_answer(solution, allow_fallback=False)
            finished = bool((token_ids == eos_id).any().item())
            records.append(
                {
                    "id": row["id"],
                    "candidate_index": candidate_index,
                    "question": row["question"],
                    "answer": row["answer"],
                    "solution": solution,
                    "parsed_answer": parsed,
                    "parser_method": method,
                    "answer_match": parsed == row["answer"],
                    "finish_status": "eos" if finished else "max_tokens",
                }
            )
    return records


def summarize(path, selected_ids, model_id):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["id"] in selected_ids]
    matched = sum(row["answer_match"].casefold() == "true" for row in rows)
    covered = {
        row["id"] for row in rows if row["answer_match"].casefold() == "true"
    }
    summary = {
        "model": model_id,
        "questions": len(selected_ids),
        "candidate_rows": len(rows),
        "answer_matched_candidates": matched,
        "questions_with_matched_candidate": len(covered),
        "coverage": len(covered) / len(selected_ids) if selected_ids else 0.0,
    }
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    validate_args(args)
    rows = read_math_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]
    selected_ids = {row["id"] for row in rows}
    source_by_id = {row["id"]: row for row in rows}

    ensure_parent(args.output)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or --overwrite")

    completed = load_completed(args.output, source_by_id) if args.resume else set()
    pending = [
        row
        for row in rows
        if any(
            (row["id"], index) not in completed
            for index in range(args.candidates_per_question)
        )
    ]
    if not pending:
        summarize(args.output, selected_ids, args.model)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = require_cuda()
    model, tokenizer = load_model(args.model, dtype)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            generated = generate_batch(model, tokenizer, batch, args)
            for record in generated:
                key = (record["id"], int(record["candidate_index"]))
                if key not in completed:
                    writer.writerow(record)
                    completed.add(key)
            file.flush()
            print(f"Progress: {min(start + len(batch), len(pending))}/{len(pending)}")
    summarize(args.output, selected_ids, args.model)


if __name__ == "__main__":
    main()
