# Qwen2.5 Math Baseline

Zero-shot inference and exact-match evaluation for the Deep Learning Challenge
2026 math dataset using `Qwen/Qwen2.5-3B-Instruct`.

## Colab setup

```bash
!git clone https://github.com/OWNER/math-algorithm.git
%cd math-algorithm
!pip install -r requirements.txt
```

Place the competition CSV files in `deep-learning-challenge-2026/`. Competition
data and generated predictions are intentionally excluded from Git.

## Evaluate a reproducible 100-row train sample

```bash
!python baseline_inference.py --limit 100 --batch-size 4
```

The default output is:

```text
deep-learning-challenge-2026/baseline_train_predictions.csv
```

The output includes the model response, extracted integer prediction, ground
truth answer, and exact-match result.

## Run leaderboard inference

```bash
!python baseline_inference.py \
  --input deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv \
  --output deep-learning-challenge-2026/baseline_leaderboard_predictions.csv \
  --limit 0 \
  --batch-size 4 \
  --resume
```

Use `--resume` to continue an interrupted run. Use `--overwrite` to replace an
existing output file.
