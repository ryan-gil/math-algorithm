import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from deepseek_runtime import generate_prompts, load_model
from reasoning_common import (
    DEFAULT_TRAIN, DEFAULT_WORK_DIR, VERIFIER_MODEL_ID, read_math_csv, solution_sha256,
)


DEFAULT_CANDIDATES = DEFAULT_WORK_DIR / "candidates_unique.csv"
DEFAULT_INDEPENDENT = DEFAULT_WORK_DIR / "independent_solutions.csv"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "deepseek_candidate_judgments.csv"
FIELDS = [
    "id", "candidate_index", "solution_sha256", "independent_solution_sha256",
    "verdict", "failure_type", "reason", "parse_status", "raw_judgment",
]
SYSTEM_PROMPT = (
    "You are a strict verifier of mathematical training solutions. An independent solution "
    "was produced before seeing the candidates. Check each candidate for interpretation errors, "
    "unsupported steps, arithmetic errors, circular use of the reference answer, and whether its "
    "final integer follows from its work. Return exactly one JSON object with key 'judgments'. "
    "Each item must contain candidate_index, verdict (PASS, FAIL, or UNCERTAIN), failure_type "
    "(NONE, INTERPRETATION, REASONING, CALCULATION, FINAL_ANSWER, INCOMPLETE, or OTHER), and a "
    "short plain-text reason. Do not omit any candidate."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Judge all unique candidates in one call per question.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--independent", type=Path, default=DEFAULT_INDEPENDENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=VERIFIER_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_json(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        judgments = value.get("judgments") if isinstance(value, dict) else None
        if isinstance(judgments, list):
            return judgments, "ok"
    return [], "parse_error"


def prompt_text(tokenizer, source, independent, candidates):
    blocks = []
    for row in candidates:
        blocks.append(
            f"[Candidate {row['candidate_index']}]\n{row['solution']}"
        )
    user = (
        f"Problem:\n{source['question']}\n\n"
        f"Reference integer label:\n{source['answer']}\n\n"
        f"Independent solution (created without the label or candidates):\n"
        f"{independent['independent_solution']}\n\n"
        "Candidates:\n" + "\n\n".join(blocks)
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}],
        tokenize=False,
        add_generation_prompt=True,
    ) + "<think>\n"


def load_completed(path, grouped, independent):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected judgment columns: {reader.fieldnames}")
        output_rows = list(reader)
    by_id = defaultdict(list)
    seen = set()
    for row in output_rows:
        key = (row["id"], row["candidate_index"])
        if key in seen:
            raise ValueError(f"Duplicate judgment row: {key}")
        seen.add(key)
        by_id[row["id"]].append(row)

    completed = set()
    for problem_id, rows in by_id.items():
        if problem_id not in grouped or problem_id not in independent:
            raise ValueError(f"Stale judgment question: {problem_id}")
        expected = {row["candidate_index"]: row for row in grouped[problem_id]}
        actual = {row["candidate_index"]: row for row in rows}
        if set(actual) != set(expected):
            raise ValueError(f"Partial or stale candidate judgments: {problem_id}")
        independent_hash = solution_sha256(independent[problem_id]["independent_solution"])
        for candidate_index, row in actual.items():
            if row["solution_sha256"] != solution_sha256(expected[candidate_index]["solution"]):
                raise ValueError(f"Candidate changed after judgment: {problem_id}/{candidate_index}")
            if row["independent_solution_sha256"] != independent_hash:
                raise ValueError(f"Independent solution changed after judgment: {problem_id}")
        completed.add(problem_id)
    return completed


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    source = {row["id"]: row for row in read_math_csv(args.source)}
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    with args.independent.open("r", encoding="utf-8-sig", newline="") as handle:
        independent = {row["id"]: row for row in csv.DictReader(handle)}

    grouped = defaultdict(list)
    for row in candidate_rows:
        grouped[row["id"]].append(row)
    eligible_ids = [
        problem_id for problem_id in grouped
        if problem_id in independent
        and independent[problem_id]["parsed_answer"] == source[problem_id]["answer"]
        and independent[problem_id]["parser_method"] == "final_answer"
        and independent[problem_id]["finish_status"] == "eos"
    ]
    if args.limit:
        eligible_ids = eligible_ids[: args.limit]
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}; use --resume or --overwrite")
    completed = load_completed(args.output, grouped, independent) if args.resume else set()
    pending = [problem_id for problem_id in eligible_ids if problem_id not in completed]
    if not pending:
        print(f"All {len(eligible_ids)} candidate bundles are complete")
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
            batch_ids = pending[start : start + args.batch_size]
            prompts = [
                prompt_text(tokenizer, source[problem_id], independent[problem_id], grouped[problem_id])
                for problem_id in batch_ids
            ]
            outputs = generate_prompts(
                model, tokenizer, prompts, args.max_input_tokens, args.max_new_tokens
            )
            for problem_id, (raw, finished) in zip(batch_ids, outputs):
                parsed, parse_status = parse_json(raw)
                by_index = {str(item.get("candidate_index")): item for item in parsed}
                independent_hash = solution_sha256(
                    independent[problem_id]["independent_solution"]
                )
                for candidate_position, candidate in enumerate(grouped[problem_id]):
                    item = by_index.get(candidate["candidate_index"], {})
                    verdict = str(item.get("verdict", "UNCERTAIN")).upper()
                    failure_type = str(item.get("failure_type", "OTHER")).upper()
                    reason = str(item.get("reason", "Missing candidate judgment.")).strip()
                    item_ok = verdict in {"PASS", "FAIL", "UNCERTAIN"} and bool(reason)
                    writer.writerow(
                        {
                            "id": problem_id,
                            "candidate_index": candidate["candidate_index"],
                            "solution_sha256": solution_sha256(candidate["solution"]),
                            "independent_solution_sha256": independent_hash,
                            "verdict": verdict if item_ok else "UNCERTAIN",
                            "failure_type": failure_type if item_ok else "OTHER",
                            "reason": reason if item_ok else "Invalid verifier item.",
                            "parse_status": (
                                "ok" if parse_status == "ok" and item_ok and finished
                                else "incomplete_or_parse_error"
                            ),
                            "raw_judgment": raw if candidate_position == 0 else "",
                        }
                    )
            handle.flush()
            print(f"Progress: {min(start + len(batch_ids), len(pending))}/{len(pending)}")

    with args.output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        "eligible_questions": len(eligible_ids),
        "judgment_rows": len(rows),
        "verdict_counts": {
            verdict: sum(row["verdict"] == verdict for row in rows)
            for verdict in ("PASS", "FAIL", "UNCERTAIN")
        },
        "parse_errors": sum(row["parse_status"] != "ok" for row in rows),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
