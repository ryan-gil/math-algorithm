import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("deep-learning-challenge-2026/deep_chal_math_train.csv")
DEFAULT_TRAIN_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_train_90.csv"
)
DEFAULT_VALID_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_valid_10.csv"
)
DEFAULT_REPORT_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_split_report.json"
)

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?!\w)")
LEADING_LABEL_PATTERN = re.compile(
    r"^\s*(?:(?:problem|question)\s+)?(?:\d+|[ivxlcdm]+)[.):]\s*",
    re.IGNORECASE,
)
LEADING_SCORE_PATTERN = re.compile(r"^\s*\[\s*\d+\s*\]\s*")
SPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"<num>|<url>|[a-z]+|\\[a-z]+|[+*/=^<>-]")

TRANSLATION_NOISE_PATTERNS = (
    re.compile(r"translate\s+(?:the\s+)?(?:above\s+)?(?:text|problem)", re.I),
    re.compile(r"output\s+the\s+translation", re.I),
    re.compile(r"translation\s+(?:is|was)\s+(?:provided|made)", re.I),
    re.compile(r"text\s+has\s+been\s+translated", re.I),
    re.compile(r"please\s+note\s+that\s+the\s+translation", re.I),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a leakage-aware train/validation split for math SFT."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--valid-output", type=Path, default=DEFAULT_VALID_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-candidates",
        type=int,
        default=200,
        help="Number of deterministic seeds evaluated for distribution balance.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames != ["id", "question", "answer"]:
        raise ValueError(
            f"Expected columns ['id', 'question', 'answer'], got {fieldnames}"
        )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, fieldnames


def remove_translation_boilerplate(text):
    kept_lines = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in TRANSLATION_NOISE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def template_fingerprint(question):
    text = remove_translation_boilerplate(question).casefold()
    text = LEADING_LABEL_PATTERN.sub("", text)
    text = LEADING_SCORE_PATTERN.sub("", text)
    text = URL_PATTERN.sub(" <url> ", text)
    text = NUMBER_PATTERN.sub(" <num> ", text)
    tokens = TOKEN_PATTERN.findall(text)
    normalized = " ".join(tokens)
    return normalized or SPACE_PATTERN.sub(" ", text).strip()


def question_category(question):
    text = question.casefold()
    categories = (
        (
            "probability_combinatorics",
            ("probability", "expected", "random", "ways", "permutation", "combination"),
        ),
        (
            "number_theory",
            ("prime", "divisible", "remainder", "modulo", "gcd", "lcm", "digit"),
        ),
        (
            "geometry",
            ("triangle", "circle", "radius", "diameter", "angle", "perimeter", "area", "volume"),
        ),
        (
            "algebra",
            ("equation", "polynomial", "function", "root", "sequence", "integer values"),
        ),
        (
            "applied_arithmetic",
            ("price", "cost", "percent", "rate", "distance", "minutes", "hours", "money"),
        ),
    )
    for category, keywords in categories:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def length_bucket(question):
    length = len(question)
    if length <= 128:
        return "000-128"
    if length <= 256:
        return "129-256"
    if length <= 512:
        return "257-512"
    return "513+"


def answer_bucket(answer):
    value = int(answer)
    if value < 0:
        sign = "negative"
    elif value == 0:
        sign = "zero"
    else:
        sign = "positive"

    digits = len(str(abs(value)))
    if digits >= 5:
        digit_bucket = "5+digits"
    else:
        digit_bucket = f"{digits}digit"
    return f"{sign}:{digit_bucket}"


def row_features(row):
    question = row["question"]
    has_visual = bool(
        re.search(r"!\[|\[asy\]|https?://\S+\.(?:png|jpe?g|gif)", question, re.I)
    )
    has_translation_noise = any(
        pattern.search(question) for pattern in TRANSLATION_NOISE_PATTERNS
    )
    return (
        ("category", question_category(question)),
        ("length", length_bucket(question)),
        ("answer", answer_bucket(row["answer"])),
        ("visual", str(has_visual)),
        ("translation_noise", str(has_translation_noise)),
    )


def build_groups(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[template_fingerprint(row["question"])].append(index)
    return dict(groups)


def candidate_validation_groups(groups, target_rows, seed):
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    selected = set()
    selected_rows = 0
    for fingerprint, indices in group_items:
        group_size = len(indices)
        current_distance = abs(target_rows - selected_rows)
        added_distance = abs(target_rows - (selected_rows + group_size))

        if selected_rows < target_rows and (
            selected_rows + group_size <= target_rows
            or added_distance < current_distance
        ):
            selected.add(fingerprint)
            selected_rows += group_size

        if selected_rows == target_rows:
            break

    return selected


def feature_counts(rows, indices):
    counts = Counter()
    for index in indices:
        counts.update(row_features(rows[index]))
    return counts


def balance_score(rows, valid_indices, total_feature_counts):
    valid_counts = feature_counts(rows, valid_indices)
    total_rows = len(rows)
    valid_rows = len(valid_indices)
    differences = []

    for feature, total_count in total_feature_counts.items():
        if total_count < 5:
            continue
        overall_share = total_count / total_rows
        valid_share = valid_counts[feature] / valid_rows
        differences.append(abs(overall_share - valid_share))

    return max(differences, default=0.0) + sum(differences) / max(len(differences), 1)


def choose_split(rows, groups, valid_ratio, base_seed, seed_candidates):
    target_rows = round(len(rows) * valid_ratio)
    all_indices = range(len(rows))
    total_feature_counts = feature_counts(rows, all_indices)
    best = None

    for offset in range(seed_candidates):
        seed = base_seed + offset
        valid_groups = candidate_validation_groups(groups, target_rows, seed)
        valid_indices = sorted(
            index
            for fingerprint in valid_groups
            for index in groups[fingerprint]
        )
        score = balance_score(rows, valid_indices, total_feature_counts)
        candidate = (score, abs(len(valid_indices) - target_rows), seed, valid_groups)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    _, _, selected_seed, valid_groups = best
    valid_indices = {
        index
        for fingerprint in valid_groups
        for index in groups[fingerprint]
    }
    train_indices = [index for index in range(len(rows)) if index not in valid_indices]
    return train_indices, sorted(valid_indices), selected_seed


def write_rows(path, rows, fieldnames, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution(rows):
    counts = feature_counts(rows, range(len(rows)))
    result = defaultdict(dict)
    for (feature_name, feature_value), count in sorted(counts.items()):
        result[feature_name][feature_value] = {
            "count": count,
            "ratio": round(count / len(rows), 6),
        }
    return dict(result)


def validate_split(source_rows, train_rows, valid_rows):
    source_ids = [row["id"] for row in source_rows]
    train_ids = {row["id"] for row in train_rows}
    valid_ids = {row["id"] for row in valid_rows}

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source dataset contains duplicate IDs")
    if train_ids.intersection(valid_ids):
        raise ValueError("Train and validation IDs overlap")
    if train_ids.union(valid_ids) != set(source_ids):
        raise ValueError("Split contains missing or unexpected IDs")

    train_fingerprints = {
        template_fingerprint(row["question"]) for row in train_rows
    }
    valid_fingerprints = {
        template_fingerprint(row["question"]) for row in valid_rows
    }
    if train_fingerprints.intersection(valid_fingerprints):
        raise ValueError("A normalized question template crosses the split boundary")


def main():
    args = parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.seed_candidates < 1:
        raise ValueError("--seed-candidates must be at least 1")

    rows, fieldnames = read_rows(args.input)
    groups = build_groups(rows)
    train_indices, valid_indices, selected_seed = choose_split(
        rows,
        groups,
        args.valid_ratio,
        args.seed,
        args.seed_candidates,
    )
    train_rows = [rows[index] for index in train_indices]
    valid_rows = [rows[index] for index in valid_indices]
    validate_split(rows, train_rows, valid_rows)

    write_rows(args.train_output, train_rows, fieldnames, args.overwrite)
    write_rows(args.valid_output, valid_rows, fieldnames, args.overwrite)

    repeated_groups = [indices for indices in groups.values() if len(indices) > 1]
    report = {
        "source": str(args.input),
        "source_rows": len(rows),
        "train_output": str(args.train_output),
        "train_rows": len(train_rows),
        "valid_output": str(args.valid_output),
        "valid_rows": len(valid_rows),
        "requested_valid_ratio": args.valid_ratio,
        "actual_valid_ratio": round(len(valid_rows) / len(rows), 8),
        "base_seed": args.seed,
        "evaluated_seed_candidates": args.seed_candidates,
        "selected_seed": selected_seed,
        "template_groups": len(groups),
        "repeated_template_groups": len(repeated_groups),
        "rows_in_repeated_template_groups": sum(map(len, repeated_groups)),
        "train_distribution": distribution(train_rows),
        "valid_distribution": distribution(valid_rows),
        "checks": {
            "id_overlap": 0,
            "missing_ids": 0,
            "template_group_overlap": 0,
        },
    }

    if args.report_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.report_output}. Use --overwrite."
        )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
