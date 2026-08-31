import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from deepseek_runtime import generate_prompts, load_model
from independent_partition import select_partition
from reasoning_common import (
    DEFAULT_TRAIN, DEFAULT_WORK_DIR, VERIFIER_MODEL_ID, extract_final_answer,
    read_math_csv, solution_sha256,
)


DEFAULT_INPUT = DEFAULT_WORK_DIR / "candidates_unique.csv"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "independent_solutions.csv"
FIELDS = [
    "id", "question_sha256", "model", "independent_solution", "parsed_answer",
    "parser_method", "finish_status",
]
SYSTEM_PROMPT = (
    "Solve the mathematics problem independently. You are not given a reference answer or "
    "another solution. Check the interpretation and calculations. Keep the derivation concise "
    "and end with exactly one line 'Final answer: <integer>'."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Independently solve candidate-covered questions.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=VERIFIER_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--partition", choices=("all", "colab", "local"), default="all",
        help="Stable workload partition; use local for this machine's 30%% share.",
    )
    parser.add_argument(
        "--colab-percent", type=int, default=70,
        help="Percentage assigned to the colab partition (default: 70).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_completed(path, source):
    completed = {}
    if not path.exists() or path.stat().st_size == 0:
        return completed
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected output columns: {reader.fieldnames}")
        for row in reader:
            expected = solution_sha256(source[row["id"]]["question"])
            if row["question_sha256"] != expected:
                raise ValueError(f"Question changed after independent solve: {row['id']}")
            completed[row["id"]] = row
    return completed


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    source = {row["id"]: row for row in read_math_csv(args.source)}
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    ids = list(dict.fromkeys(row["id"] for row in candidate_rows))
    ids = select_partition(ids, args.partition, args.colab_percent)
    if args.limit:
        ids = ids[: args.limit]
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or --overwrite")
    completed = load_completed(args.output, source) if args.resume else {}
    pending = [source[problem_id] for problem_id in ids if problem_id not in completed]
    if not pending:
        print(f"All {len(ids)} independent solutions are complete")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\nProblem:\n{row['question']}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) + "<think>\n"
                for row in batch
            ]
            outputs = generate_prompts(
                model, tokenizer, prompts, args.max_input_tokens, args.max_new_tokens
            )
            for row, (text, finished) in zip(batch, outputs):
                solution = text.strip()
                parsed, method = extract_final_answer(solution, allow_fallback=False)
                writer.writerow(
                    {
                        "id": row["id"],
                        "question_sha256": solution_sha256(row["question"]),
                        "model": args.model,
                        "independent_solution": solution,
                        "parsed_answer": parsed,
                        "parser_method": method,
                        "finish_status": "eos" if finished else "max_tokens",
                    }
                )
            handle.flush()
            print(f"Progress: {min(start + len(batch), len(pending))}/{len(pending)}")

    with args.output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["id"] in set(ids)]
    summary = {
        "model": args.model,
        "questions": len(ids),
        "completed": len(selected),
        "parsed": sum(bool(row["parsed_answer"]) for row in selected),
        "finished": sum(row["finish_status"] == "eos" for row in selected),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
