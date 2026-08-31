import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import (
    DEFAULT_TRAIN, DEFAULT_WORK_DIR, read_math_csv, solution_sha256, write_csv,
)


FIELDS = ["id", "question", "solution", "answer", "completion", "solution_source", "candidate_index"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build SFT only from independently verified solutions.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_DIR / "candidates_unique.csv")
    parser.add_argument("--independent", type=Path, default=DEFAULT_WORK_DIR / "independent_solutions.csv")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_WORK_DIR / "deepseek_candidate_judgments.csv")
    parser.add_argument("--sympy", type=Path, default=DEFAULT_WORK_DIR / "sympy_crosschecks.csv")
    parser.add_argument("--rejudge-queue", type=Path, default=DEFAULT_WORK_DIR / "rejudge_queue.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "reasoning_train_sft.csv")
    parser.add_argument("--min-solution-chars", type=int, default=80)
    parser.add_argument("--max-solution-chars", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_score(row):
    solution = row["solution"].strip()
    body = re.split(r"(?im)^\s*Final\s+answer\s*:", solution, maxsplit=1)[0]
    equation_cues = len(re.findall(r"[=+*/^]|\\(?:frac|sqrt|times|cdot)", body))
    target_length = 700
    return (
        min(equation_cues, 20),
        max(0, 1500 - abs(len(solution) - target_length)),
        -int(row["candidate_index"]),
    )


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
    source_rows = read_math_csv(args.source)
    source = {row["id"]: row for row in source_rows}
    candidates = {
        (row["id"], row["candidate_index"]): row for row in read_rows(args.candidates)
    }
    independent = {row["id"]: row for row in read_rows(args.independent)}
    judgments = read_rows(args.judgments)
    sympy_status = {
        (row["id"], row["candidate_index"]): row["status"] for row in read_rows(args.sympy)
    }
    rejudge_ids = {row["id"] for row in read_rows(args.rejudge_queue)}

    accepted = defaultdict(list)
    rejected = Counter()
    for judgment in judgments:
        key = (judgment["id"], judgment["candidate_index"])
        candidate = candidates.get(key)
        if candidate is None:
            rejected["missing_candidate"] += 1
            continue
        if candidate["id"] in rejudge_ids:
            rejected["rejudge_queue"] += 1
            continue
        independent_row = independent.get(candidate["id"])
        if not independent_row or independent_row["parsed_answer"] != source[candidate["id"]]["answer"]:
            rejected["independent_not_confirmed"] += 1
            continue
        if judgment["parse_status"] != "ok" or judgment["verdict"] != "PASS":
            rejected["deepseek_not_pass"] += 1
            continue
        if judgment["solution_sha256"] != solution_sha256(candidate["solution"]):
            rejected["solution_hash_mismatch"] += 1
            continue
        if judgment["independent_solution_sha256"] != solution_sha256(
            independent_row["independent_solution"]
        ):
            rejected["independent_hash_mismatch"] += 1
            continue
        if sympy_status.get(key) == "FAIL":
            rejected["sympy_failure"] += 1
            continue
        solution = candidate["solution"].strip()
        if not args.min_solution_chars <= len(solution) <= args.max_solution_chars:
            rejected["solution_length"] += 1
            continue
        accepted[candidate["id"]].append(candidate)

    output_rows = []
    for row in source_rows:
        options = accepted[row["id"]]
        if not options:
            continue
        chosen = max(options, key=candidate_score)
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "solution": chosen["solution"].strip(),
                "answer": row["answer"],
                "completion": chosen["solution"].strip(),
                "solution_source": "qwen25-7b_deepseek14b_verified",
                "candidate_index": chosen["candidate_index"],
            }
        )

    write_csv(args.output, FIELDS, output_rows)
    summary = {
        "source_rows": len(source_rows),
        "sft_rows": len(output_rows),
        "reasoning_coverage": len(output_rows) / len(source_rows),
        "rejudge_excluded": len(rejudge_ids),
        "rejected_candidates": dict(rejected),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
