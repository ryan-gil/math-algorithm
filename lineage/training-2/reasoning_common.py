import csv
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "deep-learning-challenge-2026" / "train-valid_split(2)"
DEFAULT_TRAIN = DATA_DIR / "deep_chal_math_train_semantic_llm_90.csv"
DEFAULT_VALID = DATA_DIR / "deep_chal_math_valid_semantic_llm_10.csv"

TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

GENERATION_SYSTEM_PROMPT = (
    "You are a precise mathematics teacher. Solve the problem independently. "
    "Write a concise, logically valid derivation using equations where useful. "
    "End with exactly one line in the form 'Final answer: <integer>'."
)
STUDENT_SYSTEM_PROMPT = (
    "You are a precise math solver. Give a concise step-by-step solution. "
    "End with exactly one line in the form 'Final answer: <integer>'."
)

FINAL_ANSWER_PATTERN = re.compile(
    r"(?im)^\s*final\s+answer\s*:\s*\$?\s*"
    r"(?P<value>[+-]?\d[\d,]*(?:\.0+)?)\s*\$?\s*[.!]?\s*$"
)
BOXED_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(?P<value>[+-]?\d[\d,]*(?:\.0+)?)\s*\}"
)
INTEGER_PATTERN = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.0+)?(?![\w.])")


def normalize_integer(value):
    value = str(value).strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+", value):
        return str(int(value))
    if re.fullmatch(r"[+-]?\d+\.0+", value):
        return str(int(value.split(".", maxsplit=1)[0]))
    return ""


def extract_final_answer(text, allow_fallback=True):
    final_matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if final_matches:
        return normalize_integer(final_matches[-1].group("value")), "final_answer"

    boxed_matches = list(BOXED_PATTERN.finditer(text))
    if boxed_matches:
        return normalize_integer(boxed_matches[-1].group("value")), "boxed"

    if allow_fallback:
        integer_matches = INTEGER_PATTERN.findall(text)
        if integer_matches:
            return normalize_integer(integer_matches[-1]), "last_integer"
    return "", "none"


def read_math_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {"id", "question", "answer"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    seen = set()
    for line_number, row in enumerate(rows, start=2):
        row_id = row["id"].strip()
        question = row["question"].strip()
        answer = normalize_integer(row["answer"])
        if not row_id or not question or not answer:
            raise ValueError(f"Invalid row in {path} at CSV line {line_number}")
        if row_id in seen:
            raise ValueError(f"Duplicate ID in {path}: {row_id}")
        seen.add(row_id)
        row["id"] = row_id
        row["question"] = question
        row["answer"] = answer
    return rows


def teacher_messages(question):
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def student_prompt_messages(question):
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)

