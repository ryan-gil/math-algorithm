import csv
import re
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are a precise math solver. Solve the problem internally and return "
    "only the final integer. Do not include an explanation or extra text."
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w)")


def read_math_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {"id", "question", "answer"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    seen_ids = set()
    for row_number, row in enumerate(rows, start=2):
        row_id = row["id"].strip()
        question = row["question"].strip()
        answer = normalize_integer(row["answer"])
        if not row_id or not question or not answer:
            raise ValueError(f"Empty or invalid value in {path} at CSV row {row_number}")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate ID in {path}: {row_id}")
        seen_ids.add(row_id)
        row["id"] = row_id
        row["question"] = question
        row["answer"] = answer

    return rows


def prompt_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def training_example(row):
    return {
        "prompt": prompt_messages(row["question"]),
        "completion": [{"role": "assistant", "content": row["answer"]}],
    }


def extract_integer(text):
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return ""
    return normalize_integer(matches[-1])


def normalize_integer(value):
    value = str(value).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+", value):
        return str(int(value))
    if re.fullmatch(r"[+-]?\d+\.0+", value):
        return str(int(value.split(".", maxsplit=1)[0]))
    return ""
