import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import (
    DEFAULT_TRAIN, DEFAULT_WORK_DIR, extract_final_answer, read_math_csv, write_csv,
)


DEFAULT_INPUT = SCRIPT_DIR.parent / "outputs(2)" / "reasoning" / "solution_candidates.csv"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "candidates_reconciled.csv"
FIELDS = [
    "id", "candidate_index", "question", "answer", "solution", "parsed_answer",
    "parser_method", "answer_match", "finish_status", "original_answer", "label_changed",
]
ISSUE_FIELDS = ["id", "issue", "old_answer", "current_answer", "candidate_rows"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconcile existing teacher candidates with the current cleaned labels."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--issues", type=Path, default=DEFAULT_WORK_DIR / "candidate_reconcile_issues.csv"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.output, args.issues):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite")
    source = {row["id"]: row for row in read_math_csv(args.source)}
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    required = {
        "id", "candidate_index", "question", "answer", "solution", "parsed_answer",
        "parser_method", "answer_match", "finish_status",
    }
    if not candidates or not required.issubset(candidates[0]):
        raise ValueError(f"Candidate input is empty or has an invalid schema: {args.input}")

    keys = set()
    output_rows = []
    issues = {}
    counters = Counter()
    for row in candidates:
        key = (row["id"], row["candidate_index"])
        if key in keys:
            raise ValueError(f"Duplicate candidate key: {key}")
        keys.add(key)
        current = source.get(row["id"])
        if current is None:
            counters["deleted_source"] += 1
            issue = issues.setdefault(
                row["id"],
                {"id": row["id"], "issue": "DELETED_SOURCE", "old_answer": row["answer"],
                 "current_answer": "", "candidate_rows": 0},
            )
            issue["candidate_rows"] += 1
            continue
        if row["question"].strip() != current["question"]:
            counters["question_changed"] += 1
            issue = issues.setdefault(
                row["id"],
                {"id": row["id"], "issue": "QUESTION_CHANGED_REGENERATE",
                 "old_answer": row["answer"], "current_answer": current["answer"],
                 "candidate_rows": 0},
            )
            issue["candidate_rows"] += 1
            continue

        parsed, method = extract_final_answer(row["solution"], allow_fallback=False)
        label_changed = row["answer"].strip() != current["answer"]
        output_rows.append(
            {
                "id": row["id"],
                "candidate_index": row["candidate_index"],
                "question": current["question"],
                "answer": current["answer"],
                "solution": row["solution"],
                "parsed_answer": parsed,
                "parser_method": method,
                "answer_match": str(bool(parsed) and parsed == current["answer"]),
                "finish_status": row["finish_status"],
                "original_answer": row["answer"].strip(),
                "label_changed": str(label_changed),
            }
        )
        counters["label_changed_rows" if label_changed else "unchanged_rows"] += 1

    write_csv(args.output, FIELDS, output_rows)
    write_csv(args.issues, ISSUE_FIELDS, list(issues.values()))
    summary = {
        "input_candidate_rows": len(candidates),
        "reconciled_candidate_rows": len(output_rows),
        "issue_questions": len(issues),
        **dict(counters),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
