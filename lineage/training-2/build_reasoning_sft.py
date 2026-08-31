import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import DEFAULT_TRAIN, PROJECT_DIR, ensure_parent, read_math_csv


DEFAULT_CANDIDATES = PROJECT_DIR / "outputs(2)" / "reasoning" / "solution_candidates.csv"
DEFAULT_JUDGMENTS = PROJECT_DIR / "outputs(2)" / "reasoning" / "solution_judgments.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs(2)" / "reasoning" / "reasoning_train_sft.csv"
FIELDS = ["id", "question", "solution", "answer", "completion", "solution_source", "candidate_index"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter answer-matched teacher solutions into a reasoning SFT dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-solution-chars", type=int, default=80)
    parser.add_argument("--max-solution-chars", type=int, default=5000)
    parser.add_argument("--min-matching-candidates", type=int, default=1)
    parser.add_argument(
        "--missing-policy",
        choices=("final-only", "drop", "error"),
        default="final-only",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def candidate_score(row):
    solution = row["solution"].strip()
    body = re.split(r"(?im)^\s*Final\s+answer\s*:", solution, maxsplit=1)[0]
    equation_cues = len(re.findall(r"[=+*/^]|\\(?:frac|sqrt|times|cdot)", body))
    target_length = 700
    length_score = max(0, 1500 - abs(len(solution) - target_length))
    return (
        row["finish_status"] == "eos",
        min(equation_cues, 20),
        length_score,
        -int(row["candidate_index"]),
    )


def rejection_reason(row, args):
    solution = row["solution"].strip()
    if row["answer_match"].casefold() != "true":
        return "answer_mismatch"
    if row["parser_method"] != "final_answer":
        return "missing_final_marker"
    if row["finish_status"] != "eos":
        return "truncated"
    if len(solution) < args.min_solution_chars:
        return "too_short"
    if len(solution) > args.max_solution_chars:
        return "too_long"
    if len(re.findall(r"(?im)^\s*final\s+answer\s*:", solution)) != 1:
        return "multiple_final_markers"
    return ""


def read_candidates(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "id", "candidate_index", "question", "answer", "solution",
        "parsed_answer", "parser_method", "answer_match", "finish_status",
    }
    if not rows:
        raise ValueError(f"No candidate rows found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    return rows


def read_judgments(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "id", "candidate_index", "solution_sha256", "verdict", "reason", "parse_status"
    }
    if not rows:
        raise ValueError(f"No judgment rows found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing judgment columns: {sorted(missing)}")
    judgments = {}
    for row in rows:
        key = (row["id"], row["candidate_index"])
        if key in judgments:
            raise ValueError(f"Duplicate judgment key: {key}")
        judgments[key] = row
    return judgments


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
    source_rows = read_math_csv(args.source)
    source_by_id = {row["id"]: row for row in source_rows}
    candidates = read_candidates(args.candidates)
    judgments = read_judgments(args.judgments)

    accepted = defaultdict(list)
    rejected = Counter()
    seen_solutions = defaultdict(set)
    for row in candidates:
        if row["id"] not in source_by_id:
            rejected["unknown_id"] += 1
            continue
        source = source_by_id[row["id"]]
        if row["question"].strip() != source["question"] or row["answer"].strip() != source["answer"]:
            rejected["source_mismatch"] += 1
            continue
        reason = rejection_reason(row, args)
        if reason:
            rejected[reason] += 1
            continue
        judgment = judgments.get((row["id"], row["candidate_index"]))
        if judgment is None:
            rejected["missing_judgment"] += 1
            continue
        if judgment["parse_status"] != "ok":
            rejected["judgment_parse_error"] += 1
            continue
        expected_hash = hashlib.sha256(row["solution"].encode("utf-8")).hexdigest()
        if judgment["solution_sha256"] != expected_hash:
            rejected["judgment_solution_mismatch"] += 1
            continue
        if judgment["verdict"] != "PASS":
            rejected["judge_fail"] += 1
            continue
        normalized = normalize_text(row["solution"])
        if normalized in seen_solutions[row["id"]]:
            rejected["duplicate_solution"] += 1
            continue
        seen_solutions[row["id"]].add(normalized)
        accepted[row["id"]].append(row)

    output_rows = []
    missing_ids = []
    for source in source_rows:
        options = accepted[source["id"]]
        if len(options) >= args.min_matching_candidates:
            chosen = max(options, key=candidate_score)
            solution = chosen["solution"].strip()
            output_rows.append(
                {
                    "id": source["id"],
                    "question": source["question"],
                    "solution": solution,
                    "answer": source["answer"],
                    "completion": solution,
                    "solution_source": "qwen25-7b-verified",
                    "candidate_index": chosen["candidate_index"],
                }
            )
            continue

        missing_ids.append(source["id"])
        if args.missing_policy == "final-only":
            completion = f"Final answer: {source['answer']}"
            output_rows.append(
                {
                    "id": source["id"],
                    "question": source["question"],
                    "solution": "",
                    "answer": source["answer"],
                    "completion": completion,
                    "solution_source": "final-only-fallback",
                    "candidate_index": "",
                }
            )
        elif args.missing_policy == "error":
            raise RuntimeError(
                f"No accepted solution for {len(missing_ids)} rows; first ID: {missing_ids[0]}"
            )

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    reasoning_rows = sum(row["solution_source"] == "qwen25-7b-verified" for row in output_rows)
    summary = {
        "source_rows": len(source_rows),
        "candidate_rows": len(candidates),
        "sft_rows": len(output_rows),
        "reasoning_rows": reasoning_rows,
        "final_only_rows": sum(row["solution_source"] == "final-only-fallback" for row in output_rows),
        "dropped_rows": len(source_rows) - len(output_rows),
        "reasoning_coverage": reasoning_rows / len(source_rows),
        "rejected_candidates": dict(rejected),
        "missing_solution_examples": missing_ids[:20],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
