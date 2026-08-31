# Qwen2.5 Reasoning SFT Pipeline

This folder is independent from `training(1)`. It uses the cleaned files in
`deep-learning-challenge-2026/train-valid_split(2)`.

## Models

- Teacher for solution generation: `Qwen/Qwen2.5-7B-Instruct`
- Student for QLoRA training: `Qwen/Qwen2.5-3B-Instruct`

During solution generation, the teacher never receives the gold answer.
Generated solutions are retained as candidates only when their explicit
`Final answer: <integer>` matches the train label. The separate verifier sees
the reference answer so it can reject invalid intermediate reasoning.
Validation labels are used only for exact-match evaluation.

## Install

Run from the repository root in a CUDA environment:

```bash
python -m pip install -r "training(2)/requirements-training.txt"
```

## Recommended smoke test

Generate solutions for 20 rows before starting the full run:

```bash
python "training(2)/generate_solutions.py" \
  --limit 20 \
  --candidates-per-question 2 \
  --output "outputs(2)/smoke/solution_candidates.csv"

python "training(2)/judge_solutions.py" \
  --input "outputs(2)/smoke/solution_candidates.csv" \
  --output "outputs(2)/smoke/solution_judgments.csv"

python "training(2)/build_reasoning_sft.py" \
  --source "deep-learning-challenge-2026/train-valid_split(2)/deep_chal_math_train_semantic_llm_90.csv" \
  --candidates "outputs(2)/smoke/solution_candidates.csv" \
  --judgments "outputs(2)/smoke/solution_judgments.csv" \
  --output "outputs(2)/smoke/reasoning_train_sft.csv" \
  --missing-policy drop \
  --overwrite
```

Inspect the generated CSV and its summary before the full run.

## Full automatic run

```bash
python "training(2)/run_reasoning_pipeline.py" --stage all
```

The default performs four sampled 7B attempts per train question, builds the
7B verifier pass, builds the SFT dataset, trains the 3B student for epoch 1,
evaluates it, resumes through epoch 2, evaluates again, and records the best
adapter.

Full 7B generation is the longest stage. It can be interrupted and resumed by
running the same command. Candidate rows are flushed after every batch.

For a smaller first full experiment:

```bash
python "training(2)/run_reasoning_pipeline.py" \
  --stage all \
  --candidates-per-question 2
```

If GPU memory is insufficient, keep generation batch size at 1 and reduce the
training batch size while retaining gradient accumulation:

```bash
python "training(2)/run_reasoning_pipeline.py" \
  --stage all \
  --generation-batch-size 1 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 32
```

## Run stages separately

```bash
python "training(2)/run_reasoning_pipeline.py" --stage generate
python "training(2)/run_reasoning_pipeline.py" --stage judge
python "training(2)/run_reasoning_pipeline.py" --stage build
python "training(2)/run_reasoning_pipeline.py" --stage train
```

The build summary reports reasoning coverage. The default `final-only` policy
retains questions for which the teacher produced no accepted rationale as
`Final answer: <label>`. Use `--missing-policy drop` to train only on verified
reasoning rows, or `error` to require complete coverage.

## Outputs

Default outputs are written under `outputs(2)/reasoning`:

- `solution_candidates.csv`: every teacher candidate and answer/parser checks
- `solution_judgments.csv`: strict 7B verification of answer-matched candidates
- `reasoning_train_sft.csv`: one selected completion per train question
- `student-qlora/adapter-epoch-1`: epoch 1 adapter
- `student-qlora/adapter-epoch-2`: epoch 2 adapter
- `student-qlora/valid-epoch-*.csv`: validation reasoning and predictions
- `student-qlora/pipeline-summary.json`: best epoch and metrics

Sample 50 validation failures after training:

```bash
python "training(2)/sample_reasoning_errors.py" --overwrite
```

## Important checks

- Review `reasoning_train_sft.summary.json` before training.
- Prefer high reasoning coverage, but do not accept truncated or mismatched solutions.
- Exact-match accuracy is the model-selection metric; training loss is secondary.
- `max_length=2048` and `max_new_tokens=512` are initial settings. Increase only
  if the summaries show substantial truncation and the GPU has enough memory.
- Training stops if any prompt plus completion exceeds `max_length`, preventing
  silent removal of the final answer. Increase `--max-length` or shorten the
  generated solutions. `--allow-truncation` is available only for reviewed runs.
