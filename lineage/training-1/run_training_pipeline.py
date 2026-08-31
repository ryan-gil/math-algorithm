import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TRAIN = PROJECT_DIR / (
    "deep-learning-challenge-2026/train-valid_split(1)/deep_chal_math_train_90.csv"
)
DEFAULT_VALID = PROJECT_DIR / (
    "deep-learning-challenge-2026/train-valid_split(1)/deep_chal_math_valid_10.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/qwen25-3b-qlora"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run epoch-1 training/evaluation, resume to epoch 2, evaluate again, "
            "and record the best adapter."
        )
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def display_path(path):
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def run(command):
    print("\n$ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_DIR,
        check=True,
    )


def complete_checkpoints(output_dir):
    return [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]


def train_epoch(args, epoch):
    adapter_dir = args.output_dir / f"adapter-epoch-{epoch}"
    run_summary = args.output_dir / f"run-summary-epoch-{epoch}.json"
    if adapter_dir.exists() and run_summary.exists():
        print(f"Epoch {epoch} training already complete: {display_path(adapter_dir)}")
        return adapter_dir

    command = [
        sys.executable,
        SCRIPT_DIR / "train_qlora.py",
        "--train-file",
        args.train_file,
        "--valid-file",
        args.valid_file,
        "--output-dir",
        args.output_dir,
        "--target-epochs",
        epoch,
        "--train-batch-size",
        args.train_batch_size,
        "--eval-batch-size",
        args.eval_batch_size,
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        "--max-length",
        args.max_length,
        "--max-train-samples",
        args.max_train_samples,
        "--max-eval-samples",
        args.max_eval_samples,
    ]
    checkpoints = complete_checkpoints(args.output_dir)
    if epoch == 2 or checkpoints:
        command.extend(["--resume-from-checkpoint", "latest"])
    run(command)
    return adapter_dir


def evaluate_epoch(args, epoch, adapter_dir):
    output_csv = args.output_dir / f"valid-epoch-{epoch}.csv"
    summary_path = output_csv.with_suffix(".summary.json")
    if summary_path.exists():
        print(f"Epoch {epoch} validation already complete: {display_path(summary_path)}")
    else:
        command = [
            sys.executable,
            SCRIPT_DIR / "evaluate_qlora.py",
            "--adapter-path",
            adapter_dir,
            "--input",
            args.valid_file,
            "--output",
            output_csv,
            "--batch-size",
            args.eval_batch_size,
            "--max-new-tokens",
            args.max_new_tokens,
            "--limit",
            args.max_eval_samples,
        ]
        if output_csv.exists():
            command.append("--resume")
        run(command)

    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def trainer_eval_loss(output_dir, epoch):
    path = output_dir / f"run-summary-epoch-{epoch}.json"
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return summary.get("eval_metrics", {}).get("eval_loss")


def write_pipeline_summary(args, results):
    best_epoch = max(
        results,
        key=lambda epoch: (
            results[epoch]["exact_match_accuracy"],
            -epoch,
        ),
    )
    summary = {
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "epochs": {
            str(epoch): {
                "adapter_path": str(args.output_dir / f"adapter-epoch-{epoch}"),
                "trainer_eval_loss": trainer_eval_loss(args.output_dir, epoch),
                **results[epoch],
            }
            for epoch in sorted(results)
        },
        "best_epoch": best_epoch,
        "best_adapter_path": str(args.output_dir / f"adapter-epoch-{best_epoch}"),
        "selection_metric": "validation exact_match_accuracy",
    }
    summary_path = args.output_dir / "pipeline-summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nValidation comparison")
    for epoch in sorted(results):
        accuracy = results[epoch]["exact_match_accuracy"]
        loss = summary["epochs"][str(epoch)]["trainer_eval_loss"]
        print(f"Epoch {epoch}: exact match={accuracy:.4%}, eval loss={loss}")
    print(f"Best adapter: {summary['best_adapter_path']}")
    print(f"Pipeline summary: {summary_path}")


def main():
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(args.train_file)
    if not args.valid_file.exists():
        raise FileNotFoundError(args.valid_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for epoch in (1, 2):
        print(f"\n{'=' * 18} EPOCH {epoch} {'=' * 18}")
        adapter_dir = train_epoch(args, epoch)
        results[epoch] = evaluate_epoch(args, epoch, adapter_dir)
        accuracy = results[epoch]["exact_match_accuracy"]
        print(f"Epoch {epoch} validation exact match: {accuracy:.4%}")

    write_pipeline_summary(args, results)


if __name__ == "__main__":
    main()
