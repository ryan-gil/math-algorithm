import argparse
import csv
import json
import re
import sys
from pathlib import Path

import sympy


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import DEFAULT_WORK_DIR, write_csv


DEFAULT_CANDIDATES = DEFAULT_WORK_DIR / "candidates_unique.csv"
DEFAULT_JUDGMENTS = DEFAULT_WORK_DIR / "deepseek_candidate_judgments.csv"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "sympy_crosschecks.csv"
FIELDS = ["id", "candidate_index", "status", "checked_equalities", "failure"]
ALLOWED = re.compile(r"^[0-9+*/().\-\s^]+$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservatively check purely numeric equalities in passed solutions."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_expression(value):
    value = value.replace("$", "").replace(",", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.strip().rstrip(".,;:").replace("^", "**")
    if not value or not ALLOWED.fullmatch(value.replace("**", "^")):
        return None
    try:
        expression = sympy.sympify(value, evaluate=True)
    except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return None
    return expression if not expression.free_symbols else None


def check_solution(solution):
    checked = 0
    for line in solution.splitlines():
        if "=" not in line:
            continue
        parts = [clean_expression(part) for part in line.split("=")]
        for left, right in zip(parts, parts[1:]):
            if left is None or right is None:
                continue
            checked += 1
            if sympy.simplify(left - right) != 0:
                return "FAIL", checked, line.strip()
    return ("PASS", checked, "") if checked else ("NOT_APPLICABLE", 0, "")


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        candidates = {
            (row["id"], row["candidate_index"]): row for row in csv.DictReader(handle)
        }
    with args.judgments.open("r", encoding="utf-8-sig", newline="") as handle:
        judgments = list(csv.DictReader(handle))
    output_rows = []
    for judgment in judgments:
        if judgment["verdict"] != "PASS" or judgment["parse_status"] != "ok":
            continue
        key = (judgment["id"], judgment["candidate_index"])
        status, checked, failure = check_solution(candidates[key]["solution"])
        output_rows.append(
            {
                "id": key[0],
                "candidate_index": key[1],
                "status": status,
                "checked_equalities": checked,
                "failure": failure,
            }
        )
    write_csv(args.output, FIELDS, output_rows)
    summary = {
        "candidate_rows": len(output_rows),
        "pass": sum(row["status"] == "PASS" for row in output_rows),
        "fail": sum(row["status"] == "FAIL" for row in output_rows),
        "not_applicable": sum(row["status"] == "NOT_APPLICABLE" for row in output_rows),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
