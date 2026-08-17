# Qwen2.5 Math Baseline

Zero-shot inference and exact-match evaluation for the Deep Learning Challenge
2026 math dataset using `Qwen/Qwen2.5-3B-Instruct`.

## Colab setup

```bash
!git clone https://github.com/ryan-gil/math-algorithm.git
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

## QLoRA training in Colab

Use a GPU runtime and mount Google Drive. The model is fixed to
`Qwen/Qwen2.5-3B-Instruct`; only LoRA adapter weights are trained.

```python
!pip install -q -r requirements-training.txt

from google.colab import drive
drive.mount("/content/drive")
```

Put the semantic train/valid CSV files in
`/content/drive/MyDrive/math_algorithm/data/`, then train the first epoch. The
output directory is on Drive so Trainer checkpoints survive a runtime reset.

```bash
!python train_qlora.py \
  --train-file "/content/drive/MyDrive/math_algorithm/data/deep_chal_math_train_semantic_90.csv" \
  --valid-file "/content/drive/MyDrive/math_algorithm/data/deep_chal_math_valid_semantic_10.csv" \
  --output-dir "/content/drive/MyDrive/math_algorithm/qwen25-3b-qlora" \
  --target-epochs 1
```

Evaluate the saved epoch-1 adapter with exact-match accuracy:

```bash
!python evaluate_qlora.py \
  --adapter-path "/content/drive/MyDrive/math_algorithm/qwen25-3b-qlora/adapter-epoch-1" \
  --input "/content/drive/MyDrive/math_algorithm/data/deep_chal_math_valid_semantic_10.csv" \
  --output "/content/drive/MyDrive/math_algorithm/qwen25-3b-qlora/valid-epoch-1.csv" \
  --batch-size 4
```

If epoch 1 improves validation accuracy, resume its full Trainer checkpoint and
train to a total of two epochs. This adds one epoch, not two more epochs.
The first run already uses a two-epoch learning-rate schedule and stops at the
epoch-1 boundary, so optimizer and scheduler state can be resumed correctly.

```bash
!python train_qlora.py \
  --train-file "/content/drive/MyDrive/math_algorithm/data/deep_chal_math_train_semantic_90.csv" \
  --valid-file "/content/drive/MyDrive/math_algorithm/data/deep_chal_math_valid_semantic_10.csv" \
  --output-dir "/content/drive/MyDrive/math_algorithm/qwen25-3b-qlora" \
  --target-epochs 2 \
  --resume-from-checkpoint latest
```

Evaluate `adapter-epoch-2` with the same command after changing the adapter and
output paths. Use `--max-train-samples 64 --max-eval-samples 32` with a new
output directory for a short pipeline smoke test before the full run.

## Rejudge the top 1,000 SAME_TOPIC_ONLY candidates in Colab

Select a T4 GPU runtime, install the judging dependencies, and mount Google
Drive so checkpoints survive a Colab disconnect.

```python
!pip install -q -r requirements-llm-judge.txt

from google.colab import drive
drive.mount("/content/drive")
```

Upload `deep_chal_math_similarity_candidates.csv` to a Drive directory, then
run:

```bash
!python llm_rejudge_candidates.py \
  --input "/content/drive/MyDrive/math_algorithm/deep_chal_math_similarity_candidates.csv" \
  --output "/content/drive/MyDrive/math_algorithm/deep_chal_math_llm_rejudged_same_topic_only_top1000.csv" \
  --summary "/content/drive/MyDrive/math_algorithm/deep_chal_math_llm_rejudged_same_topic_only_top1000_summary.json" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --source-labels SAME_TOPIC_ONLY \
  --top-n 1000 \
  --selection-mode combined \
  --batch-size 2
```

The selection ranks `SAME_TOPIC_ONLY` pairs by `min(S_raw, S_mask)`. Rerun the
same command to resume. Add
`--retry-parse-errors` to retry malformed model responses. Use `--overwrite`
only when intentionally starting over.
