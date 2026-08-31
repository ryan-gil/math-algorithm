import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import DEFAULT_TRAIN, DEFAULT_WORK_DIR, read_math_csv, write_csv


FIELDS = [
    "id", "question", "label_answer", "reason_codes", "independent_answer",
    "reconciled_candidates", "unique_matching_candidates", "deepseek_pass",
    "deepseek_fail", "deepseek_uncertain", "sympy_fail", "recommended_next_action",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect question-level cases for later rejudging.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--reconciled", type=Path, default=DEFAULT_WORK_DIR / "candidates_reconciled.csv")
    parser.add_argument("--unique", type=Path, default=DEFAULT_WORK_DIR / "candidates_unique.csv")
    parser.add_argument("--independent", type=Path, default=DEFAULT_WORK_DIR / "independent_solutions.csv")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_WORK_DIR / "deepseek_candidate_judgments.csv")
    parser.add_argument("--sympy", type=Path, default=DEFAULT_WORK_DIR / "sympy_crosschecks.csv")
    parser.add_argument("--reconcile-issues", type=Path, default=DEFAULT_WORK_DIR / "candidate_reconcile_issues.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "rejudge_queue.csv")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
    source = {row["id"]: row for row in read_math_csv(args.source)}
    reconciled = rows(args.reconciled)
    unique = rows(args.unique)
    independent = {row["id"]: row for row in rows(args.independent)}
    judgments = rows(args.judgments)
    sympy_rows = rows(args.sympy)
    issues = rows(args.reconcile_issues)

    reconciled_count = Counter(row["id"] for row in reconciled)
    unique_count = Counter(row["id"] for row in unique)
    judgment_counts = defaultdict(Counter)
    for row in judgments:
        verdict = row["verdict"] if row["parse_status"] == "ok" else "UNCERTAIN"
        judgment_counts[row["id"]][verdict] += 1
    sympy_fail = Counter(row["id"] for row in sympy_rows if row["status"] == "FAIL")
    issue_codes = defaultdict(set)
    for row in issues:
        if row["id"] in source:
            issue_codes[row["id"]].add(row["issue"])

    output_rows = []
    reason_counts = Counter()
    for problem_id, problem in source.items():
        reasons = set(issue_codes[problem_id])
        if reconciled_count[problem_id] == 0:
            reasons.add("NO_RECONCILED_CANDIDATE")
        if unique_count[problem_id] == 0:
            reasons.add("NO_UNIQUE_ANSWER_MATCHED_CANDIDATE")
        independent_row = independent.get(problem_id)
        if unique_count[problem_id] and independent_row is None:
            reasons.add("MISSING_INDEPENDENT_SOLUTION")
        elif independent_row:
            if independent_row["finish_status"] != "eos":
                reasons.add("INDEPENDENT_TRUNCATED")
            if independent_row["parser_method"] != "final_answer":
                reasons.add("INDEPENDENT_PARSE_FAILURE")
            if independent_row["parsed_answer"] != problem["answer"]:
                reasons.add("INDEPENDENT_LABEL_DISAGREEMENT")

        counts = judgment_counts[problem_id]
        usable_passes = max(0, counts["PASS"] - sympy_fail[problem_id])
        if unique_count[problem_id] and independent_row and not reasons.intersection(
            {"INDEPENDENT_TRUNCATED", "INDEPENDENT_PARSE_FAILURE", "INDEPENDENT_LABEL_DISAGREEMENT"}
        ):
            if sum(counts.values()) == 0:
                reasons.add("MISSING_DEEPSEEK_JUDGMENT")
            elif usable_passes == 0:
                reasons.add("NO_CANDIDATE_MATCHES_VERIFIER_DIRECTION_AND_ANSWER")
        if sympy_fail[problem_id] and usable_passes == 0:
            reasons.add("SYMPY_NUMERIC_EQUALITY_FAILURE")
        if not reasons:
            continue

        for reason in reasons:
            reason_counts[reason] += 1
        label_issue = bool(
            reasons.intersection(
                {
                    "INDEPENDENT_LABEL_DISAGREEMENT", "INDEPENDENT_PARSE_FAILURE",
                    "QUESTION_CHANGED_REGENERATE",
                }
            )
        )
        action = (
            "REJUDGE_LABEL_AND_QUESTION"
            if label_issue
            else "REJUDGE_THEN_REGENERATE_CANDIDATES_IF_UNRESOLVED"
        )
        output_rows.append(
            {
                "id": problem_id,
                "question": problem["question"],
                "label_answer": problem["answer"],
                "reason_codes": "|".join(sorted(reasons)),
                "independent_answer": independent_row["parsed_answer"] if independent_row else "",
                "reconciled_candidates": reconciled_count[problem_id],
                "unique_matching_candidates": unique_count[problem_id],
                "deepseek_pass": counts["PASS"],
                "deepseek_fail": counts["FAIL"],
                "deepseek_uncertain": counts["UNCERTAIN"],
                "sympy_fail": sympy_fail[problem_id],
                "recommended_next_action": action,
            }
        )

    write_csv(args.output, FIELDS, output_rows)
    summary = {
        "source_questions": len(source),
        "rejudge_questions": len(output_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "action_counts": dict(Counter(row["recommended_next_action"] for row in output_rows)),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
