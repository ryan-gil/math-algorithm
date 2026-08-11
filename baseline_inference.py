import argparse
import csv
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT = Path("deep-learning-challenge-2026/deep_chal_math_train.csv")
DEFAULT_OUTPUT = Path("deep-learning-challenge-2026/baseline_train_predictions.csv")
SYSTEM_PROMPT = (
    "You are a precise math solver. Solve the problem internally and return "
    "only the final integer. Do not include an explanation or extra text."
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a zero-shot Qwen2.5 math baseline on a CSV file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of reproducibly sampled rows. Use 0 for every row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs that already exist in the output CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV.",
    )
    return parser.parse_args()


def read_input(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {"id", "question"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return rows


def select_rows(rows, limit, seed):
    if limit < 0:
        raise ValueError("--limit must be 0 or a positive integer")
    if limit == 0 or limit >= len(rows):
        return rows

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in selected_indices]


def resolve_device(requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    return requested_device


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def format_prompts(tokenizer, rows):
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def generate_batch(model, tokenizer, rows, max_new_tokens):
    prompts = format_prompts(tokenizer, rows)
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def extract_integer(text):
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return ""
    value = matches[-1].replace(",", "")
    if "." in value:
        integer_part, fractional_part = value.split(".", maxsplit=1)
        if not fractional_part or set(fractional_part) != {"0"}:
            return ""
        value = integer_part
    try:
        return str(int(value))
    except ValueError:
        return ""


def normalize_answer(value):
    value = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+", value):
        return value
    return str(int(value))


def output_fields(has_answers):
    fields = ["id", "prediction", "raw_response"]
    if has_answers:
        fields.extend(["answer", "correct"])
    return fields


def read_completed_ids(path, expected_fields):
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Existing output columns {reader.fieldnames} do not match "
                f"expected columns {expected_fields}"
            )
        return {row["id"] for row in reader}


def summarize_output(path, selected_ids, has_answers):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["id"] in selected_ids]

    valid_predictions = sum(bool(row["prediction"]) for row in rows)
    print(f"Completed: {len(rows)}/{len(selected_ids)}")
    print(f"Valid integer predictions: {valid_predictions}/{len(rows)}")
    if has_answers and rows:
        correct = sum(row["correct"].lower() == "true" for row in rows)
        print(f"Exact-match accuracy: {correct}/{len(rows)} = {correct / len(rows):.2%}")


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume and --overwrite")

    rows = read_input(args.input)
    selected_rows = select_rows(rows, args.limit, args.seed)
    selected_ids = {row["id"] for row in selected_rows}
    has_answers = "answer" in rows[0]
    fields = output_fields(has_answers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --resume or --overwrite."
        )
    if args.output.exists() and args.overwrite:
        args.output.unlink()

    completed_ids = read_completed_ids(args.output, fields) if args.resume else set()
    pending_rows = [row for row in selected_rows if row["id"] not in completed_ids]

    print(f"Input: {args.input}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Already completed: {len(selected_rows) - len(pending_rows)}")
    print(f"Output: {args.output}")

    if not pending_rows:
        summarize_output(args.output, selected_ids, has_answers)
        return

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device == "cpu":
        print("Warning: Qwen2.5-3B inference on CPU can be very slow.")

    model, tokenizer = load_model(args.model, device)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    processed = len(selected_rows) - len(pending_rows)
    running_correct = 0
    running_scored = 0

    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for start in range(0, len(pending_rows), args.batch_size):
            batch = pending_rows[start : start + args.batch_size]
            responses = generate_batch(
                model,
                tokenizer,
                batch,
                args.max_new_tokens,
            )

            for row, response in zip(batch, responses):
                prediction = extract_integer(response)
                result = {
                    "id": row["id"],
                    "prediction": prediction,
                    "raw_response": response.strip(),
                }
                if has_answers:
                    answer = normalize_answer(row["answer"])
                    is_correct = prediction == answer
                    result.update({"answer": answer, "correct": is_correct})
                    running_correct += int(is_correct)
                    running_scored += 1
                writer.writerow(result)

            file.flush()
            processed += len(batch)
            message = f"Progress: {processed}/{len(selected_rows)}"
            if running_scored:
                message += f" | current accuracy: {running_correct / running_scored:.2%}"
            print(message, flush=True)

    summarize_output(args.output, selected_ids, has_answers)


if __name__ == "__main__":
    main()
