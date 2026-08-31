import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_solutions import FIELDS as CANDIDATE_FIELDS
from reasoning_common import PROJECT_DIR, TEACHER_MODEL_ID, ensure_parent


DEFAULT_INPUT = PROJECT_DIR / "outputs(2)" / "reasoning" / "solution_candidates.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs(2)" / "reasoning" / "solution_judgments.csv"
FIELDS = [
    "id",
    "candidate_index",
    "solution_sha256",
    "verdict",
    "reason",
    "parse_status",
    "raw_judgment",
]
SYSTEM_PROMPT = (
    "You are a strict verifier of mathematical solutions. Check whether every "
    "important step is logically valid and whether the solution reaches the "
    "provided reference integer. Return one JSON object only, with keys "
    "verdict and reason. verdict must be PASS or FAIL. Do not use LaTeX in reason."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge answer-matched solution candidates with Qwen2.5-7B-Instruct."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=TEACHER_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_candidates(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CANDIDATE_FIELDS:
            raise ValueError(f"Unexpected candidate columns: {reader.fieldnames}")
        rows = list(reader)
    return [
        row
        for row in rows
        if row["answer_match"].casefold() == "true"
        and row["parser_method"] == "final_answer"
        and row["finish_status"] == "eos"
    ]


def solution_sha256(solution):
    return hashlib.sha256(solution.encode("utf-8")).hexdigest()


def load_completed(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected judgment columns: {reader.fieldnames}")
        completed = {}
        for row in reader:
            key = (row["id"], row["candidate_index"])
            if key in completed:
                raise ValueError(f"Duplicate judgment key: {key}")
            completed[key] = row["solution_sha256"]
        return completed


def parse_json_object(text):
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            verdict = str(value.get("verdict", "")).strip().upper()
            reason = str(value.get("reason", "")).strip()
            if verdict in {"PASS", "FAIL"} and reason:
                return verdict, reason, "ok"
    return "FAIL", "Could not parse a valid verifier JSON object.", "parse_error"


def messages(row):
    user = (
        "Problem:\n" + row["question"]
        + "\n\nReference final answer:\n" + row["answer"]
        + "\n\nCandidate solution:\n" + row["solution"]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def load_model(model_id):
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit Qwen2.5-7B judging requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def judge_batch(model, tokenizer, rows, args):
    prompts = [
        tokenizer.apply_chat_template(
            messages(row), tokenize=False, add_generation_prompt=True
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
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def summarize(path, eligible_count):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    passed = sum(row["verdict"] == "PASS" and row["parse_status"] == "ok" for row in rows)
    summary = {
        "eligible_candidates": eligible_count,
        "judged_candidates": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "parse_errors": sum(row["parse_status"] != "ok" for row in rows),
        "pass_rate": passed / len(rows) if rows else 0.0,
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")
    candidates = read_candidates(args.input)
    if args.limit:
        candidates = candidates[: args.limit]
    ensure_parent(args.output)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or --overwrite")
    completed = load_completed(args.output) if args.resume else {}
    candidate_by_key = {
        (row["id"], row["candidate_index"]): row for row in candidates
    }
    for key, saved_hash in completed.items():
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        if saved_hash != solution_sha256(candidate["solution"]):
            raise ValueError(f"Candidate changed after judgment for {key}; do not resume")
    pending = [
        row for row in candidates
        if (row["id"], row["candidate_index"]) not in completed
    ]
    if not pending:
        summarize(args.output, len(candidates))
        return

    model, tokenizer = load_model(args.model)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            responses = judge_batch(model, tokenizer, batch, args)
            for row, response in zip(batch, responses):
                verdict, reason, status = parse_json_object(response)
                writer.writerow(
                    {
                        "id": row["id"],
                        "candidate_index": row["candidate_index"],
                        "solution_sha256": solution_sha256(row["solution"]),
                        "verdict": verdict,
                        "reason": reason,
                        "parse_status": status,
                        "raw_judgment": response.strip(),
                    }
                )
            file.flush()
            print(f"Progress: {min(start + len(batch), len(pending))}/{len(pending)}")
    summarize(args.output, len(candidates))


if __name__ == "__main__":
    main()
