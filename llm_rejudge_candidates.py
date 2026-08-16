import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_INPUT = Path(
    "deep-learning-challenge-2026/deep_chal_math_similarity_candidates.csv"
)
DEFAULT_OUTPUT = Path(
    "deep-learning-challenge-2026/"
    "deep_chal_math_llm_rejudged_same_topic_only_top1000.csv"
)
DEFAULT_SUMMARY = Path(
    "deep-learning-challenge-2026/"
    "deep_chal_math_llm_rejudged_same_topic_only_top1000_summary.json"
)
ALLOWED_SOURCE_LABELS = {"SAME_REASONING", "SAME_TOPIC_ONLY"}
ALLOWED_LLM_LABELS = {
    "SAME_TEMPLATE",
    "SAME_REASONING",
    "SAME_TOPIC_ONLY",
    "UNRELATED",
}
OUTPUT_FIELDS = (
    "rank",
    "left_id",
    "right_id",
    "source_label",
    "raw_similarity",
    "masked_similarity",
    "token_jaccard",
    "ranking_score",
    "llm_label",
    "same_unknown",
    "same_equation_skeleton",
    "same_operation_sequence",
    "question_1_unknown",
    "question_2_unknown",
    "question_1_equation",
    "question_2_equation",
    "confidence",
    "reason",
    "parse_status",
    "raw_response",
    "left_question",
    "right_question",
)

SYSTEM_PROMPT = """You are a careful judge of whether two math word problems are structurally equivalent.

Classify the pair into exactly one label:
- SAME_TEMPLATE: The same problem template. Only numbers, names, units, or superficial wording differ. The roles of the givens, the unknown, constraints, and solution steps are the same.
- SAME_REASONING: The wording or context differs, but the same unknown is obtained with the same equation skeleton and operation sequence.
- SAME_TOPIC_ONLY: The topic or vocabulary overlaps, but the unknown, condition direction, equation skeleton, or required operation sequence differs.
- UNRELATED: They do not share a meaningful mathematical structure.

Important rules:
- Compare mathematical roles, not numerical answers.
- Treat inverse questions as SAME_TOPIC_ONLY, for example final = original*(1-rate) versus original = final/(1-rate).
- Carefully distinguish at least/at most, maximum/minimum, increase/decrease, area/perimeter, with/without replacement, and permutations/combinations.
- Do not assume two problems have the same reasoning merely because they use the same objects or topic.
- Equations must use symbolic role names instead of the original numbers when possible.

Return one JSON object only. Do not use Markdown fences or add text outside JSON.
Required schema:
{
  "label": "SAME_TEMPLATE|SAME_REASONING|SAME_TOPIC_ONLY|UNRELATED",
  "same_unknown": true,
  "same_equation_skeleton": true,
  "same_operation_sequence": true,
  "question_1_unknown": "short description",
  "question_2_unknown": "short description",
  "question_1_equation": "symbolic equation skeleton or unknown",
  "question_2_equation": "symbolic equation skeleton or unknown",
  "confidence": 0.0,
  "reason": "one concise sentence"
}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rejudge the highest-scoring embedding candidates with Qwen."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Number selected across both source labels, unless --selection-mode=per-label.",
    )
    parser.add_argument(
        "--source-labels",
        nargs="+",
        choices=sorted(ALLOWED_SOURCE_LABELS),
        default=["SAME_TOPIC_ONLY"],
        help="Candidate source labels considered before ranking.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("combined", "per-label"),
        default="combined",
        help="combined selects top-n total; per-label selects top-n from each label.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quantization",
        choices=("4bit", "none"),
        default="4bit",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Flush and fsync output after this many newly judged pairs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of resuming it.",
    )
    parser.add_argument(
        "--retry-parse-errors",
        action="store_true",
        help="On resume, retry rows whose previous parse_status was not OK.",
    )
    return parser.parse_args()


def as_float(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field!r} for pair {row.get('left_id')}/{row.get('right_id')}") from error


def pair_key(row):
    return row["left_id"], row["right_id"]


def ranking_key(row):
    raw = as_float(row, "raw_similarity")
    masked = as_float(row, "masked_similarity")
    jaccard = as_float(row, "token_jaccard")
    return (
        min(raw, masked),
        (raw + masked) / 2.0,
        jaccard,
        row["left_id"],
        row["right_id"],
    )


def ranking_score(row):
    raw = as_float(row, "raw_similarity")
    masked = as_float(row, "masked_similarity")
    return min(raw, masked)


def load_candidates(path, top_n, selection_mode, source_labels):
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    candidates = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("label") in source_labels:
                candidates.append(row)

    if selection_mode == "combined":
        selected = sorted(candidates, key=ranking_key, reverse=True)[:top_n]
    else:
        selected = []
        for label in sorted(source_labels):
            label_rows = [row for row in candidates if row["label"] == label]
            selected.extend(sorted(label_rows, key=ranking_key, reverse=True)[:top_n])
        selected.sort(key=ranking_key, reverse=True)

    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        row["ranking_score"] = ranking_score(row)
    return selected, len(candidates)


def remove_parse_error_rows(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    kept = [row for row in rows if row.get("parse_status") == "OK"]
    removed = len(rows) - len(kept)
    if not removed:
        return 0

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    return removed


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            completed.add(pair_key(row))
    return completed


def build_user_prompt(row):
    return (
        "Question 1:\n"
        f"{row['left_question']}\n\n"
        "Question 2:\n"
        f"{row['right_question']}\n\n"
        "Classify this pair using the required JSON schema."
    )


def build_chat_prompt(tokenizer, row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object found")


def normalize_bool(value, field):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"{field} must be a boolean")


def normalize_judgment(value):
    label = str(value.get("label", "")).strip().upper()
    if label not in ALLOWED_LLM_LABELS:
        raise ValueError(f"Unexpected label: {label!r}")

    confidence = float(value.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    return {
        "llm_label": label,
        "same_unknown": normalize_bool(value.get("same_unknown"), "same_unknown"),
        "same_equation_skeleton": normalize_bool(
            value.get("same_equation_skeleton"), "same_equation_skeleton"
        ),
        "same_operation_sequence": normalize_bool(
            value.get("same_operation_sequence"), "same_operation_sequence"
        ),
        "question_1_unknown": str(value.get("question_1_unknown", "")).strip(),
        "question_2_unknown": str(value.get("question_2_unknown", "")).strip(),
        "question_1_equation": str(value.get("question_1_equation", "")).strip(),
        "question_2_equation": str(value.get("question_2_equation", "")).strip(),
        "confidence": confidence,
        "reason": str(value.get("reason", "")).strip(),
    }


def parse_response(text):
    try:
        judgment = normalize_judgment(extract_json_object(text))
        return judgment, "OK"
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        empty = {
            "llm_label": "PARSE_ERROR",
            "same_unknown": "",
            "same_equation_skeleton": "",
            "same_operation_sequence": "",
            "question_1_unknown": "",
            "question_2_unknown": "",
            "question_1_equation": "",
            "question_2_equation": "",
            "confidence": "",
            "reason": "",
        }
        return empty, f"ERROR: {error}"


def load_model(model_name, quantization):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    if quantization == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit inference requires a CUDA GPU. In Colab select "
                "Runtime > Change runtime type > T4 GPU."
            )
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_batch(model, tokenizer, rows, max_input_tokens, max_new_tokens):
    prompts = [build_chat_prompt(tokenizer, row) for row in rows]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def output_row(candidate, judgment, parse_status, raw_response):
    return {
        "rank": candidate["rank"],
        "left_id": candidate["left_id"],
        "right_id": candidate["right_id"],
        "source_label": candidate["label"],
        "raw_similarity": candidate["raw_similarity"],
        "masked_similarity": candidate["masked_similarity"],
        "token_jaccard": candidate["token_jaccard"],
        "ranking_score": f'{candidate["ranking_score"]:.6f}',
        **judgment,
        "parse_status": parse_status,
        "raw_response": raw_response,
        "left_question": candidate["left_question"],
        "right_question": candidate["right_question"],
    }


def write_summary(path, args, selected, eligible_count, output_path):
    output_rows = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as file:
            output_rows = list(csv.DictReader(file))

    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "model": args.model,
        "selection_mode": args.selection_mode,
        "source_labels": args.source_labels,
        "requested_top_n": args.top_n,
        "eligible_candidates": eligible_count,
        "selected_candidates": len(selected),
        "written_rows": len(output_rows),
        "source_label_counts": dict(
            sorted(Counter(row["source_label"] for row in output_rows).items())
        ),
        "llm_label_counts": dict(
            sorted(Counter(row["llm_label"] for row in output_rows).items())
        ),
        "parse_status_counts": dict(
            sorted(Counter(row["parse_status"] for row in output_rows).items())
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main():
    args = parse_args()
    if args.top_n < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("--top-n, --batch-size, and --checkpoint-every must be positive")

    selected, eligible_count = load_candidates(
        args.input, args.top_n, args.selection_mode, set(args.source_labels)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.output.exists():
        args.output.unlink()
    removed_parse_errors = 0
    if args.retry_parse_errors and not args.overwrite:
        removed_parse_errors = remove_parse_error_rows(args.output)
    completed = load_completed(args.output)
    pending = [row for row in selected if pair_key(row) not in completed]

    print(
        json.dumps(
            {
                "eligible": eligible_count,
                "selected": len(selected),
                "already_completed": len(selected) - len(pending),
                "removed_parse_errors_for_retry": removed_parse_errors,
                "pending": len(pending),
                "selection_source_labels": dict(
                    sorted(Counter(row["label"] for row in selected).items())
                ),
            },
            indent=2,
        )
    )

    if not pending:
        write_summary(args.summary, args, selected, eligible_count, args.output)
        return

    tokenizer, model = load_model(args.model, args.quantization)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    new_rows = 0

    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()

        for start in tqdm(range(0, len(pending), args.batch_size), desc="LLM judging"):
            batch = pending[start:start + args.batch_size]
            responses = generate_batch(
                model,
                tokenizer,
                batch,
                args.max_input_tokens,
                args.max_new_tokens,
            )
            for candidate, response in zip(batch, responses):
                judgment, parse_status = parse_response(response)
                writer.writerow(
                    output_row(candidate, judgment, parse_status, response)
                )
                new_rows += 1
            if new_rows % args.checkpoint_every < len(batch):
                file.flush()
                os.fsync(file.fileno())

        file.flush()
        os.fsync(file.fileno())

    write_summary(args.summary, args, selected, eligible_count, args.output)


if __name__ == "__main__":
    main()
