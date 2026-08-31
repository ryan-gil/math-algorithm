import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reasoning_common import DEFAULT_TRAIN, DEFAULT_VALID, PROJECT_DIR, read_math_csv


DEFAULT_WORK_DIR = PROJECT_DIR / "outputs(2)" / "reasoning"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reasoning data, train two epochs, and evaluate each epoch."
    )
    parser.add_argument(
        "--stage", choices=("all", "generate", "judge", "build", "train"), default="all"
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--candidates-per-question", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--missing-policy", choices=("final-only", "drop", "error"), default="final-only")
    return parser.parse_args()


def run(command):
    print("\n$ " + shlex.join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=PROJECT_DIR, check=True)


def complete_checkpoints(output_dir):
    return [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]


def generate_solutions(args, candidates_path):
    command = [
        sys.executable,
        SCRIPT_DIR / "generate_solutions.py",
        "--input", args.train_file,
        "--output", candidates_path,
        "--candidates-per-question", args.candidates_per_question,
        "--batch-size", args.generation_batch_size,
        "--max-new-tokens", args.generation_max_new_tokens,
    ]
    if args.max_train_samples:
        command.extend(["--limit", args.max_train_samples])
    if candidates_path.exists():
        command.append("--resume")
    run(command)


def build_sft(args, candidates_path, sft_path):
    command = [
        sys.executable,
        SCRIPT_DIR / "build_reasoning_sft.py",
        "--source", args.train_file,
        "--candidates", candidates_path,
        "--judgments", args.work_dir / "solution_judgments.csv",
        "--output", sft_path,
        "--missing-policy", args.missing_policy,
        "--overwrite",
    ]
    run(command)


def judge_solutions(args, candidates_path, judgments_path):
    command = [
        sys.executable,
        SCRIPT_DIR / "judge_solutions.py",
        "--input", candidates_path,
        "--output", judgments_path,
        "--batch-size", args.generation_batch_size,
    ]
    if judgments_path.exists():
        command.append("--resume")
    run(command)


def train_epoch(args, epoch, sft_path, model_dir):
    adapter_dir = model_dir / f"adapter-epoch-{epoch}"
    summary_path = model_dir / f"run-summary-epoch-{epoch}.json"
    if adapter_dir.exists() and summary_path.exists():
        print(f"Epoch {epoch} training already complete: {adapter_dir}")
        return adapter_dir
    command = [
        sys.executable,
        SCRIPT_DIR / "train_reasoning_qlora.py",
        "--train-file", sft_path,
        "--output-dir", model_dir,
        "--target-epochs", epoch,
        "--max-length", args.max_length,
        "--train-batch-size", args.train_batch_size,
        "--gradient-accumulation-steps", args.gradient_accumulation_steps,
        "--max-train-samples", args.max_train_samples,
    ]
    if epoch == 2 or complete_checkpoints(model_dir):
        command.extend(["--resume-from-checkpoint", "latest"])
    run(command)
    return adapter_dir


def evaluate_epoch(args, epoch, adapter_dir, model_dir):
    output = model_dir / f"valid-epoch-{epoch}.csv"
    summary_path = output.with_suffix(".summary.json")
    if summary_path.exists():
        print(f"Epoch {epoch} evaluation already complete: {summary_path}")
    else:
        command = [
            sys.executable,
            SCRIPT_DIR / "evaluate_reasoning.py",
            "--adapter-path", adapter_dir,
            "--input", args.valid_file,
            "--output", output,
            "--batch-size", args.eval_batch_size,
            "--max-new-tokens", args.eval_max_new_tokens,
            "--limit", args.max_eval_samples,
        ]
        if output.exists():
            command.append("--resume")
        run(command)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_summary(args, model_dir, results):
    best_epoch = max(
        results,
        key=lambda epoch: (
            results[epoch]["exact_match_accuracy"],
            -results[epoch]["parse_failure_rate"],
            -results[epoch]["truncation_rate"],
            -epoch,
        ),
    )
    summary = {
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "epochs": {str(epoch): results[epoch] for epoch in sorted(results)},
        "best_epoch": best_epoch,
        "best_adapter_path": str(model_dir / f"adapter-epoch-{best_epoch}"),
        "selection_metric": "validation exact-match, then parse/truncation rate",
    }
    path = model_dir / "pipeline-summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nValidation comparison")
    for epoch in sorted(results):
        result = results[epoch]
        print(
            f"Epoch {epoch}: accuracy={result['exact_match_accuracy']:.4%}, "
            f"parse failures={result['parse_failure_rate']:.2%}, "
            f"truncated={result['truncation_rate']:.2%}"
        )
    print(f"Best adapter: {summary['best_adapter_path']}")


def main():
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(args.train_file)
    if not args.valid_file.exists():
        raise FileNotFoundError(args.valid_file)
    train_ids = {row["id"] for row in read_math_csv(args.train_file)}
    valid_ids = {row["id"] for row in read_math_csv(args.valid_file)}
    overlap = train_ids & valid_ids
    if overlap:
        raise ValueError(f"Train/valid IDs overlap, for example: {sorted(overlap)[:5]}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.work_dir / "solution_candidates.csv"
    judgments_path = args.work_dir / "solution_judgments.csv"
    sft_path = args.work_dir / "reasoning_train_sft.csv"
    model_dir = args.work_dir / "student-qlora"

    if args.stage in {"all", "generate"}:
        generate_solutions(args, candidates_path)
    if args.stage == "generate":
        return

    if args.stage in {"all", "judge"}:
        if not candidates_path.exists():
            raise FileNotFoundError(candidates_path)
        judge_solutions(args, candidates_path, judgments_path)
    if args.stage == "judge":
        return

    if args.stage in {"all", "build"}:
        if not candidates_path.exists():
            raise FileNotFoundError(candidates_path)
        if not judgments_path.exists():
            raise FileNotFoundError(judgments_path)
        build_sft(args, candidates_path, sft_path)
    if args.stage == "build":
        return

    if not sft_path.exists():
        raise FileNotFoundError(sft_path)
    results = {}
    for epoch in (1, 2):
        print(f"\n{'=' * 18} EPOCH {epoch} {'=' * 18}")
        adapter_dir = train_epoch(args, epoch, sft_path, model_dir)
        results[epoch] = evaluate_epoch(args, epoch, adapter_dir, model_dir)
    write_summary(args, model_dir, results)


if __name__ == "__main__":
    main()
