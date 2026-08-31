import csv
import hashlib
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "deep-learning-challenge-2026" / "train-valid_split(2)"
DEFAULT_TRAIN = DATA_DIR / "deep_chal_math_train_semantic_llm_90.csv"
DEFAULT_VALID = DATA_DIR / "deep_chal_math_valid_semantic_llm_10.csv"
DEFAULT_WORK_DIR = PROJECT_DIR / "outputs(3)" / "reasoning"

TEACHER_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
VERIFIER_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
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
        matches = INTEGER_PATTERN.findall(text)
        if matches:
            return normalize_integer(matches[-1]), "last_integer"
    return "", "none"


def read_math_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
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


def read_csv(path, required=None):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    if required:
        missing = set(required).difference(fields)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return fields, rows


def write_csv(path, fieldnames, rows):
    path = Path(path)
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def solution_sha256(solution):
    return hashlib.sha256(solution.encode("utf-8")).hexdigest()


def normalize_solution_text(text):
    text = re.sub(r"(?im)^\s*final\s+answer\s*:.*$", "", text)
    text = text.casefold().replace("\\cdot", "*").replace("\\times", "*")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9가-힣+*/^=().-]+", " ", text).strip()
