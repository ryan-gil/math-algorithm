import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_dataset import (
    LEADING_LABEL_PATTERN,
    LEADING_SCORE_PATTERN,
    NUMBER_PATTERN,
    SPACE_PATTERN,
    choose_split,
    distribution,
    read_rows,
    remove_translation_boilerplate,
    template_fingerprint,
    write_rows,
)


DEFAULT_INPUT = Path("deep-learning-challenge-2026/deep_chal_math_train.csv")
DEFAULT_TRAIN_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_train_semantic_90.csv"
)
DEFAULT_VALID_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_valid_semantic_10.csv"
)
DEFAULT_CANDIDATES_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_similarity_candidates.csv"
)
DEFAULT_GROUPS_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_similarity_groups.csv"
)
DEFAULT_REPORT_OUTPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_semantic_split_report.json"
)
DEFAULT_CACHE_DIR = Path("deep-learning-challenge-2026/embedding_cache")

WORD_PATTERN = re.compile(r"[a-z]+|<num>|\\[a-z]+|[+*/=^<>-]", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "has", "have", "how", "if", "in", "is", "it", "of",
    "on", "or", "that", "the", "then", "to", "was", "what", "when",
    "which", "with", "would",
}
CONFLICTING_PHRASES = (
    ("at least", "at most"),
    ("greater than", "less than"),
    ("maximum", "minimum"),
    ("largest", "smallest"),
    ("increase", "decrease"),
    ("with replacement", "without replacement"),
    ("clockwise", "counterclockwise"),
    ("area", "perimeter"),
)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right, max_size=None):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return True
        combined_size = self.size[left_root] + self.size[right_root]
        if max_size is not None and combined_size > max_size:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] = combined_size
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build raw/masked embedding neighbors, create conservative similar-"
            "question groups, and split them without crossing group boundaries."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--valid-output", type=Path, default=DEFAULT_VALID_OUTPUT)
    parser.add_argument(
        "--candidates-output", type=Path, default=DEFAULT_CANDIDATES_OUTPUT
    )
    parser.add_argument("--groups-output", type=Path, default=DEFAULT_GROUPS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-candidates", type=int, default=200)
    parser.add_argument("--template-mask-threshold", type=float, default=0.96)
    parser.add_argument("--template-raw-threshold", type=float, default=0.84)
    parser.add_argument("--template-jaccard-threshold", type=float, default=0.72)
    parser.add_argument("--reasoning-mask-threshold", type=float, default=0.91)
    parser.add_argument("--reasoning-raw-threshold", type=float, default=0.89)
    parser.add_argument("--reasoning-jaccard-threshold", type=float, default=0.42)
    parser.add_argument("--topic-threshold", type=float, default=0.72)
    parser.add_argument("--max-group-size", type=int, default=50)
    parser.add_argument(
        "--include-unrelated-candidates",
        action="store_true",
        help="Also write UNRELATED top-k pairs to the candidate CSV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recompute-embeddings", action="store_true")
    return parser.parse_args()


def clean_question(question, mask_numbers=False):
    text = remove_translation_boilerplate(question).casefold()
    text = LEADING_LABEL_PATTERN.sub("", text)
    text = LEADING_SCORE_PATTERN.sub("", text)
    if mask_numbers:
        text = NUMBER_PATTERN.sub(" <NUM> ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def content_tokens(text):
    return {
        token.casefold()
        for token in WORD_PATTERN.findall(text)
        if token.casefold() not in STOPWORDS
    }


def token_jaccard(left, right):
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def has_conflicting_condition(left, right):
    for first, second in CONFLICTING_PHRASES:
        left_first_only = first in left and second not in left
        left_second_only = second in left and first not in left
        right_first_only = first in right and second not in right
        right_second_only = second in right and first not in right
        if (left_first_only and right_second_only) or (
            left_second_only and right_first_only
        ):
            return True
    return False


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(input_path, model_name, view):
    payload = f"{file_sha256(input_path)}\n{model_name}\n{view}\nversion=1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def encode_questions(model, texts, cache_path, batch_size, recompute):
    if cache_path.exists() and not recompute:
        embeddings = np.load(cache_path)
        if len(embeddings) != len(texts):
            raise ValueError(f"Cached row count does not match input: {cache_path}")
        return embeddings.astype("float32", copy=False), True

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings, False


def nearest_pairs(embeddings, top_k):
    search_count = min(top_k + 1, len(embeddings))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, neighbors = index.search(embeddings, search_count)

    pairs = set()
    for left, row in enumerate(neighbors):
        added = 0
        for right in row:
            right = int(right)
            if right < 0 or right == left:
                continue
            pairs.add((min(left, right), max(left, right)))
            added += 1
            if added == top_k:
                break
    return pairs


def classify_pair(
    same_fingerprint,
    raw_similarity,
    masked_similarity,
    jaccard,
    conflicting,
    args,
):
    if same_fingerprint:
        return "SAME_TEMPLATE"

    if (
        not conflicting
        and masked_similarity >= args.template_mask_threshold
        and raw_similarity >= args.template_raw_threshold
        and jaccard >= args.template_jaccard_threshold
    ):
        return "SAME_TEMPLATE"

    if (
        not conflicting
        and masked_similarity >= args.reasoning_mask_threshold
        and raw_similarity >= args.reasoning_raw_threshold
        and jaccard >= args.reasoning_jaccard_threshold
    ):
        return "SAME_REASONING"

    if max(raw_similarity, masked_similarity) >= args.topic_threshold:
        return "SAME_TOPIC_ONLY"
    return "UNRELATED"


def build_candidate_records(rows, raw_embeddings, masked_embeddings, pairs, args):
    raw_texts = [clean_question(row["question"]) for row in rows]
    masked_texts = [clean_question(row["question"], mask_numbers=True) for row in rows]
    fingerprints = [template_fingerprint(row["question"]) for row in rows]
    records = []

    for left, right in pairs:
        raw_similarity = float(np.dot(raw_embeddings[left], raw_embeddings[right]))
        masked_similarity = float(
            np.dot(masked_embeddings[left], masked_embeddings[right])
        )
        jaccard = token_jaccard(masked_texts[left], masked_texts[right])
        conflicting = has_conflicting_condition(raw_texts[left], raw_texts[right])
        label = classify_pair(
            fingerprints[left] == fingerprints[right],
            raw_similarity,
            masked_similarity,
            jaccard,
            conflicting,
            args,
        )
        records.append(
            {
                "left_index": left,
                "right_index": right,
                "left_id": rows[left]["id"],
                "right_id": rows[right]["id"],
                "raw_similarity": raw_similarity,
                "masked_similarity": masked_similarity,
                "token_jaccard": jaccard,
                "conflicting_condition": conflicting,
                "label": label,
            }
        )

    records.sort(
        key=lambda item: (
            item["label"] not in {"SAME_TEMPLATE", "SAME_REASONING"},
            -min(item["raw_similarity"], item["masked_similarity"]),
            -max(item["raw_similarity"], item["masked_similarity"]),
        )
    )
    return records


def build_similarity_groups(rows, records, max_group_size):
    union_find = UnionFind(len(rows))
    fingerprint_first = {}
    for index, row in enumerate(rows):
        fingerprint = template_fingerprint(row["question"])
        if fingerprint in fingerprint_first:
            union_find.union(index, fingerprint_first[fingerprint])
        else:
            fingerprint_first[fingerprint] = index

    semantic_edges = [
        record
        for record in records
        if record["label"] in {"SAME_TEMPLATE", "SAME_REASONING"}
    ]
    semantic_edges.sort(
        key=lambda item: (
            min(item["raw_similarity"], item["masked_similarity"]),
            max(item["raw_similarity"], item["masked_similarity"]),
        ),
        reverse=True,
    )

    blocked_merges = 0
    for record in semantic_edges:
        merged = union_find.union(
            record["left_index"],
            record["right_index"],
            max_size=max_group_size,
        )
        if not merged:
            blocked_merges += 1

    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[union_find.find(index)].append(index)
    ordered_groups = sorted(groups.values(), key=lambda indices: min(indices))
    return {f"group_{number:05d}": indices for number, indices in enumerate(ordered_groups)}, blocked_merges


def write_candidates(path, records, rows, overwrite, include_unrelated):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "left_id", "right_id", "raw_similarity", "masked_similarity",
        "token_jaccard", "conflicting_condition", "label",
        "left_question", "right_question",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record["label"] == "UNRELATED" and not include_unrelated:
                continue
            writer.writerow(
                {
                    "left_id": record["left_id"],
                    "right_id": record["right_id"],
                    "raw_similarity": f'{record["raw_similarity"]:.6f}',
                    "masked_similarity": f'{record["masked_similarity"]:.6f}',
                    "token_jaccard": f'{record["token_jaccard"]:.6f}',
                    "conflicting_condition": record["conflicting_condition"],
                    "label": record["label"],
                    "left_question": rows[record["left_index"]]["question"],
                    "right_question": rows[record["right_index"]]["question"],
                }
            )


def write_groups(path, groups, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ("group_id", "group_size", "id", "question", "answer")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group_id, indices in groups.items():
            if len(indices) == 1:
                continue
            for index in indices:
                writer.writerow(
                    {
                        "group_id": group_id,
                        "group_size": len(indices),
                        **rows[index],
                    }
                )


def validate_group_split(rows, groups, train_indices, valid_indices):
    train_set = set(train_indices)
    valid_set = set(valid_indices)
    if train_set & valid_set:
        raise ValueError("Train and validation indices overlap")
    if train_set | valid_set != set(range(len(rows))):
        raise ValueError("Split contains missing or unexpected rows")
    crossed = [
        group_id
        for group_id, indices in groups.items()
        if train_set.intersection(indices) and valid_set.intersection(indices)
    ]
    if crossed:
        raise ValueError(f"Similarity groups cross the split: {crossed[:5]}")


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")

    rows, fieldnames = read_rows(args.input)
    raw_texts = [clean_question(row["question"]) for row in rows]
    masked_texts = [clean_question(row["question"], mask_numbers=True) for row in rows]

    model = SentenceTransformer(args.model)
    raw_cache = args.cache_dir / f"raw_{cache_key(args.input, args.model, 'raw')}.npy"
    mask_cache = args.cache_dir / f"mask_{cache_key(args.input, args.model, 'mask')}.npy"
    raw_embeddings, raw_cache_hit = encode_questions(
        model, raw_texts, raw_cache, args.batch_size, args.recompute_embeddings
    )
    masked_embeddings, mask_cache_hit = encode_questions(
        model, masked_texts, mask_cache, args.batch_size, args.recompute_embeddings
    )

    raw_pairs = nearest_pairs(raw_embeddings, args.top_k)
    masked_pairs = nearest_pairs(masked_embeddings, args.top_k)
    candidate_pairs = raw_pairs | masked_pairs
    records = build_candidate_records(
        rows, raw_embeddings, masked_embeddings, candidate_pairs, args
    )
    groups, blocked_merges = build_similarity_groups(
        rows, records, args.max_group_size
    )

    train_indices, valid_indices, selected_seed = choose_split(
        rows, groups, args.valid_ratio, args.seed, args.seed_candidates
    )
    validate_group_split(rows, groups, train_indices, valid_indices)
    train_rows = [rows[index] for index in train_indices]
    valid_rows = [rows[index] for index in valid_indices]

    write_rows(args.train_output, train_rows, fieldnames, args.overwrite)
    write_rows(args.valid_output, valid_rows, fieldnames, args.overwrite)
    write_candidates(
        args.candidates_output,
        records,
        rows,
        args.overwrite,
        args.include_unrelated_candidates,
    )
    write_groups(args.groups_output, groups, rows, args.overwrite)

    repeated_groups = [indices for indices in groups.values() if len(indices) > 1]
    label_counts = Counter(record["label"] for record in records)
    group_size_counts = Counter(len(indices) for indices in repeated_groups)
    report = {
        "source": str(args.input),
        "source_rows": len(rows),
        "model": args.model,
        "top_k_per_view": args.top_k,
        "raw_neighbor_pairs": len(raw_pairs),
        "masked_neighbor_pairs": len(masked_pairs),
        "candidate_pair_union": len(candidate_pairs),
        "candidate_labels": dict(sorted(label_counts.items())),
        "candidate_csv_excludes_unrelated": not args.include_unrelated_candidates,
        "similarity_groups": len(groups),
        "repeated_similarity_groups": len(repeated_groups),
        "rows_in_repeated_similarity_groups": sum(map(len, repeated_groups)),
        "repeated_group_size_distribution": {
            str(size): count for size, count in sorted(group_size_counts.items())
        },
        "largest_group": max(map(len, groups.values())),
        "blocked_merges_due_to_size_cap": blocked_merges,
        "train_output": str(args.train_output),
        "train_rows": len(train_rows),
        "valid_output": str(args.valid_output),
        "valid_rows": len(valid_rows),
        "actual_valid_ratio": round(len(valid_rows) / len(rows), 8),
        "selected_seed": selected_seed,
        "train_distribution": distribution(train_rows),
        "valid_distribution": distribution(valid_rows),
        "checks": {
            "id_overlap": len(
                {row["id"] for row in train_rows}
                & {row["id"] for row in valid_rows}
            ),
            "missing_ids": len(rows) - len(train_rows) - len(valid_rows),
            "similarity_group_overlap": 0,
        },
        "embedding_cache": {
            "raw": str(raw_cache),
            "masked": str(mask_cache),
            "raw_cache_hit": raw_cache_hit,
            "masked_cache_hit": mask_cache_hit,
        },
        "thresholds": {
            "template_mask": args.template_mask_threshold,
            "template_raw": args.template_raw_threshold,
            "template_jaccard": args.template_jaccard_threshold,
            "reasoning_mask": args.reasoning_mask_threshold,
            "reasoning_raw": args.reasoning_raw_threshold,
            "reasoning_jaccard": args.reasoning_jaccard_threshold,
            "topic": args.topic_threshold,
            "max_group_size": args.max_group_size,
        },
        "judge_note": (
            "Labels are conservative deterministic heuristics, not LLM judgments. "
            "Review SAME_REASONING rows before treating every edge as mathematically equivalent."
        ),
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
