import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import DEFAULT_WORK_DIR, normalize_solution_text, write_csv
from reconcile_candidates import FIELDS as RECONCILED_FIELDS


DEFAULT_INPUT = DEFAULT_WORK_DIR / "candidates_reconciled.csv"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "candidates_unique.csv"
FIELDS = RECONCILED_FIELDS + ["dedup_group", "normalized_similarity"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cheap-filter and remove near-duplicate solutions within each question."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jaccard-threshold", type=float, default=0.88)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tokens(text):
    return set(re.findall(r"[a-z0-9가-힣]+|[=+*/^-]", text))


def equation_fingerprint(text):
    normalized = normalize_solution_text(text)
    return "|".join(
        chunk.strip() for chunk in re.findall(r"[^.!?\n]*=[^.!?\n]*", normalized)
    )


def similarity(left, right):
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_duplicate(row, kept, threshold):
    normalized = normalize_solution_text(row["solution"])
    fingerprint = equation_fingerprint(row["solution"])
    for group, other, other_normalized, other_fingerprint in kept:
        if normalized == other_normalized:
            return group, 1.0
        score = similarity(normalized, other_normalized)
        same_equations = bool(fingerprint) and fingerprint == other_fingerprint
        if score >= threshold and same_equations:
            return group, score
    return None, 0.0


def main():
    args = parse_args()
    if not 0 < args.jaccard_threshold <= 1:
        raise ValueError("--jaccard-threshold must be in (0, 1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    eligible = defaultdict(list)
    rejected = Counter()
    for row in rows:
        if row["answer_match"].casefold() != "true":
            rejected["answer_mismatch"] += 1
        elif row["parser_method"] != "final_answer":
            rejected["missing_final_marker"] += 1
        elif row["finish_status"] != "eos":
            rejected["truncated"] += 1
        else:
            eligible[row["id"]].append(row)

    output_rows = []
    duplicate_rows = 0
    for problem_id, options in eligible.items():
        kept = []
        for row in sorted(options, key=lambda value: int(value["candidate_index"])):
            duplicate_group, score = is_duplicate(row, kept, args.jaccard_threshold)
            if duplicate_group is not None:
                duplicate_rows += 1
                continue
            group = len(kept)
            normalized = normalize_solution_text(row["solution"])
            fingerprint = equation_fingerprint(row["solution"])
            kept.append((group, row, normalized, fingerprint))
            output_rows.append(
                {**row, "dedup_group": group, "normalized_similarity": f"{score:.6f}"}
            )

    write_csv(args.output, FIELDS, output_rows)
    summary = {
        "input_rows": len(rows),
        "eligible_before_dedup": sum(len(value) for value in eligible.values()),
        "unique_candidate_rows": len(output_rows),
        "questions_with_candidate": len(eligible),
        "duplicate_rows_removed": duplicate_rows,
        "cheap_filter_rejections": dict(rejected),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
