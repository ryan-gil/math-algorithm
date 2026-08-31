# Reproduce outputs(3)/.../adapter-epoch-1

## 데이터 파일 정책

이 Git 저장소에는 코드와 실행 문서만 포함되며 CSV 학습 데이터와 기록된 모델 출력은
포함되지 않습니다. 학습 전에 별도로 보관된 `reasoning_train_sft.csv`를 다음 위치에
복사해야 합니다.

~~~text
data/reasoning_train_sft.csv
~~~

정확한 lineage 재구성이 필요한 경우에는 별도 보관 데이터의 디렉터리 구조를 유지해
`lineage/*/data/`와 `lineage/*/recorded/` 아래에 배치해야 합니다. 이 경로들은 실수로
원격 저장소에 업로드되지 않도록 `.gitignore`에서 제외됩니다.

## 빠른 실행 가이드 (Windows PowerShell)

이 저장소의 최종 목표는 `Qwen/Qwen2.5-3B-Instruct`를 베이스 모델로 사용해
training(3)의 `adapter-epoch-1`을 재학습하는 것입니다. 최종 학습 데이터는 이미
`data/reasoning_train_sft.csv`에 포함되어 있으므로, 모델을 만드는 데 필요한 최소
실행 파일은 `train_reasoning_qlora.py`입니다.

프로젝트 루트에서 기존 가상환경을 활성화하고 패키지를 설치합니다.

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

CUDA GPU가 PyTorch에서 보이는지 먼저 확인합니다.

~~~powershell
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
~~~

그다음 아래 명령 하나로 epoch-1 QLoRA adapter를 학습합니다.

~~~powershell
python train_reasoning_qlora.py `
  --train-file data/reasoning_train_sft.csv `
  --output-dir reproduced/student-qlora `
  --target-epochs 1 `
  --max-length 2048 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 32 `
  --max-train-samples 0
~~~

완료 후 사용할 모델 adapter는 다음 경로에 생성됩니다.

~~~text
reproduced/student-qlora/adapter-epoch-1/
~~~

베이스 모델 가중치는 Hugging Face에서 자동으로 내려받습니다. 4-bit NF4 QLoRA를
사용하지만 모델 로딩과 학습에는 CUDA GPU가 필요하며, 원 실행 환경은 VRAM 24GB의
RTX 3090이었습니다. 현재 환경에서 `torch.cuda.is_available()`가 거짓이면 CPU만으로
이 학습 스크립트를 실행할 수 없습니다.

`lineage/`는 최종 SFT가 어떻게 만들어졌는지 재검증하거나 후보 생성·DeepSeek 검증을
다시 수행할 때만 필요합니다. 단순히 최종 adapter를 학습할 때는 읽을 필요가 없습니다.

This bundle preserves the full local lineage requested for the epoch-1 adapter:

~~~text
training(1) baseline
  -> training(2) Qwen-7B candidate generation and judging
  -> DeepSeek-14B independent solving and candidate verification
  -> deterministic answer/format/dedup/SymPy/rejudge filtering
  -> 3,930-row reasoning SFT dataset
  -> training(2) QLoRA trainer
  -> outputs(3)/reasoning/student-qlora/adapter-epoch-1
~~~

The primary train_reasoning_qlora.py and reasoning_common.py at the bundle root are now
exact copies from training(2). The output(3) serialized training arguments confirm the same
training(2) settings: batch size 1, gradient accumulation 32, learning rate 5e-5, seed 42,
maximum length 2,048, and a two-epoch schedule stopped after epoch 1.

## Bundle layout

- data/reasoning_train_sft.csv: exact 3,930-row DeepSeek-filtered SFT input used by output(3).
- train_reasoning_qlora.py and reasoning_common.py: exact training(2) trainer and helper used
  to create the target adapter.
- lineage/training-1/: training(1) training/evaluation pipeline and exact train/valid data.
- lineage/training-2/: required training(2) generation, judging, SFT build, training, and
  evaluation code.
- lineage/training-2/recorded/: exact Qwen candidate, judgment, and SFT outputs from
  training(2). The candidate file is the input consumed by the DeepSeek stage.
- lineage/deepseek-filter/: DeepSeek runtime, independent inference, candidate verifier,
  reconciliation, deduplication, SymPy checking, rejudge exclusion, and final SFT builder.
- lineage/deepseek-filter/recorded/: exact expensive DeepSeek inference results and SymPy
  checks, retained so the final SFT can be rebuilt without sampling the models again.

No trained adapter weights or optimizer checkpoint is copied into this bundle: those are
outputs, not inputs.

## Recorded provenance

Models and cached revisions present for the original run:

- Student: Qwen/Qwen2.5-3B-Instruct at
  aa8e72537993ba99e69dfaafa59ed015b17504d1
- Candidate generator/judge: Qwen/Qwen2.5-7B-Instruct at
  a09a35458c702b33eeacc393d103063234e8bc28
- Independent solver/verifier: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B at
  1df8507178afcc1bef68cd8c393f61a886323761

Important artifact SHA-256 values:

~~~text
4ae55a979205e78492b82b8695c3af499a61bc2eb4f63f34bbacc95163accc9f  data/reasoning_train_sft.csv
9447acd56bb581a245f78372ec402869551e506b8388bc8574a58895134a7188  lineage/training-2/recorded/solution_candidates.csv
16a75b0904540b8ef57f389b1131ae369beedfce354e89b5a464c7c67d957e10  lineage/deepseek-filter/recorded/independent_solutions.csv
8e2f1407cbaaf8d8dd6aef80121ad05a73c1b33f549298c5247d7a54b03147b3  lineage/deepseek-filter/recorded/deepseek_candidate_judgments.csv
724d6d75a55e0772cd67d064671e5ef8c37e31fe6b1e95a5eceb9f4d1e2acb86  lineage/deepseek-filter/recorded/sympy_crosschecks.csv
3bb54f63766f15c5a24808cfccc31d6a38768fb88992766d0b416525131830c5  original adapter_model.safetensors
~~~

The final targets are selected Qwen-7B candidate solutions. DeepSeek-14B first solved each
problem independently, then verified the candidate bundles. A candidate was retained only when
the independent DeepSeek answer agreed with the current label, the DeepSeek verifier returned
PASS, its hashes matched, SymPy found no explicit numeric equality failure, and the question
was not sent to the rejudge queue. This produced 3,930 SFT rows from 14,327 source rows.

## Environment

The recorded environment was Python 3.10.20, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 92000,
and an RTX 3090 with 24 GB VRAM.

~~~bash
conda create -n adapter-e1-repro python=3.10.20 -y
conda activate adapter-e1-repro
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

If necessary, install the CUDA-specific PyTorch 2.13 build appropriate for the machine.

## Rebuild the exact final SFT dataset

The expensive, sampled inference outputs are included. The following deterministic stages
reconstruct the final SFT CSV from the recorded training(2) candidates and DeepSeek results.
Run from this bundle's root:

~~~bash
mkdir -p reproduced/filter-work

python lineage/deepseek-filter/reconcile_candidates.py \
  --source lineage/training-2/data/deep_chal_math_train_semantic_llm_90.csv \
  --input lineage/training-2/recorded/solution_candidates.csv \
  --output reproduced/filter-work/candidates_reconciled.csv \
  --issues reproduced/filter-work/candidate_reconcile_issues.csv \
  --overwrite

python lineage/deepseek-filter/deduplicate_candidates.py \
  --input reproduced/filter-work/candidates_reconciled.csv \
  --output reproduced/filter-work/candidates_unique.csv \
  --overwrite

python lineage/deepseek-filter/build_rejudge_queue.py \
  --source lineage/training-2/data/deep_chal_math_train_semantic_llm_90.csv \
  --reconciled reproduced/filter-work/candidates_reconciled.csv \
  --unique reproduced/filter-work/candidates_unique.csv \
  --independent lineage/deepseek-filter/recorded/independent_solutions.csv \
  --judgments lineage/deepseek-filter/recorded/deepseek_candidate_judgments.csv \
  --sympy lineage/deepseek-filter/recorded/sympy_crosschecks.csv \
  --reconcile-issues reproduced/filter-work/candidate_reconcile_issues.csv \
  --output reproduced/filter-work/rejudge_queue.csv \
  --overwrite

python lineage/deepseek-filter/build_reasoning_sft.py \
  --source lineage/training-2/data/deep_chal_math_train_semantic_llm_90.csv \
  --candidates reproduced/filter-work/candidates_unique.csv \
  --independent lineage/deepseek-filter/recorded/independent_solutions.csv \
  --judgments lineage/deepseek-filter/recorded/deepseek_candidate_judgments.csv \
  --sympy lineage/deepseek-filter/recorded/sympy_crosschecks.csv \
  --rejudge-queue reproduced/filter-work/rejudge_queue.csv \
  --output reproduced/filter-work/reasoning_train_sft.csv \
  --overwrite

sha256sum reproduced/filter-work/reasoning_train_sft.csv
~~~

The last hash must be 4ae55a979205e78492b82b8695c3af499a61bc2eb4f63f34bbacc95163accc9f.

## Rerun DeepSeek inference instead of using recorded results

After producing reproduced/filter-work/candidates_unique.csv above, run:

~~~bash
python lineage/deepseek-filter/solve_independently.py \
  --source lineage/training-2/data/deep_chal_math_train_semantic_llm_90.csv \
  --input reproduced/filter-work/candidates_unique.csv \
  --output reproduced/filter-work/independent_solutions.csv \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --batch-size 1 \
  --max-new-tokens 2048

python lineage/deepseek-filter/verify_candidates.py \
  --source lineage/training-2/data/deep_chal_math_train_semantic_llm_90.csv \
  --candidates reproduced/filter-work/candidates_unique.csv \
  --independent reproduced/filter-work/independent_solutions.csv \
  --output reproduced/filter-work/deepseek_candidate_judgments.csv \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --batch-size 1 \
  --max-new-tokens 2048

python lineage/deepseek-filter/sympy_crosscheck.py \
  --candidates reproduced/filter-work/candidates_unique.csv \
  --judgments reproduced/filter-work/deepseek_candidate_judgments.csv \
  --output reproduced/filter-work/sympy_crosschecks.csv \
  --overwrite
~~~

These inference calls sample at temperature 0.6/top-p 0.95, so a fresh run is not expected to
reproduce the recorded CSVs byte-for-byte. Use the recorded files for exact SFT reconstruction.

## Recreate the output(3) epoch-1 adapter

Verify CUDA, then run the exact training(2) trainer against the final filtered SFT data:

~~~bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

python train_reasoning_qlora.py \
  --train-file data/reasoning_train_sft.csv \
  --output-dir reproduced/student-qlora \
  --target-epochs 1 \
  --max-length 2048 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --max-train-samples 0
~~~

The adapter is written to reproduced/student-qlora/adapter-epoch-1/; the resumable Trainer
state is written to reproduced/student-qlora/checkpoint-123/. The original run reported loss
0.16742672837846648 and runtime 1054.1575 seconds.

GPU kernels are not guaranteed to be bitwise deterministic. Matching the code, data, package
versions, model revisions, seed, and original CUDA hardware is required for the closest possible
reproduction; an equivalent rerun may still have a different adapter SHA-256.
