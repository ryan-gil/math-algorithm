"""Stable work partitioning for independent DeepSeek generation."""

import hashlib


def _rank(problem_id):
    return hashlib.sha256(problem_id.encode("utf-8")).digest()


def colab_ids(problem_ids, colab_percent=70):
    """Return the deterministic Colab share, rounded down to an exact percentage."""
    if not 1 <= colab_percent <= 99:
        raise ValueError("colab_percent must be between 1 and 99")
    unique_ids = list(dict.fromkeys(problem_ids))
    count = len(unique_ids) * colab_percent // 100
    return set(sorted(unique_ids, key=_rank)[:count])


def select_partition(problem_ids, partition="all", colab_percent=70):
    if partition not in {"all", "colab", "local"}:
        raise ValueError(f"Unknown independent-generation partition: {partition}")
    if partition == "all":
        return list(problem_ids)
    selected_colab_ids = colab_ids(problem_ids, colab_percent)
    colab = partition == "colab"
    return [
        problem_id
        for problem_id in problem_ids
        if (problem_id in selected_colab_ids) == colab
    ]
