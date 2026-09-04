#!/bin/bash
#SBATCH --job-name=gemma_mmlu_ref
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/gemma_mmlu_ref_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/gemma_mmlu_ref_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/work/int/xuchengxue23/social_conformity_revision}"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="${ENV_PATH:-/gpfs/work/int/xuchengxue23/envs/self_history_vllm}"

MODEL_PATH="${MODEL_PATH:-$REPO/models/gemma2_2b_it}"
MODEL_NAME_LOCAL="google/gemma-2-2b-it"

DATA_PATH_LOCAL="$PROJECT/data/datasets/mmlu_all_validation.json"
EXPECTED_MD5="598fec26af1127bfd7bca06c39c1a159"

RUN_NAME="gemma2_2b_mmlu_main500_reference_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT/results/runs/$RUN_NAME/2026-08-26_self_history_ablation"

# Dynamic port so simultaneous jobs do not collide.
PORT=$((18000 + SLURM_JOB_ID % 10000))

cd "$REPO"

module purge
module load anaconda3/2022.10-gcc-8.5.0-4dp3trd
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "===== ENVIRONMENT ====="
hostname
date
nvidia-smi
python --version
python -c "import torch,vllm,transformers; print('torch=',torch.__version__,'vllm=',vllm.__version__,'transformers=',transformers.__version__)"
git branch --show-current
git log -1 --oneline

echo "===== INPUT GUARDS ====="
test -x "$ENV_PATH/bin/python"
test -d "$MODEL_PATH"
test -f "$DATA_PATH_LOCAL"

DATA_MD5=$(md5sum "$DATA_PATH_LOCAL" | awk '{print $1}')
echo "DATA_MD5=$DATA_MD5"
test "$DATA_MD5" = "$EXPECTED_MD5"

cd "$PROJECT"

export PYTHONPATH=.
export MODEL_NAME="$MODEL_NAME_LOCAL"
export API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
export DATA_PATH="$DATA_PATH_LOCAL"
export RESULT_DIR="results"
export USE_CONFIDENCE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "===== STATIC PREFLIGHT ====="
python experiments/2026-08-26_self_history_ablation/test_prompt_equivalence.py

echo "===== START VLLM ====="
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME_LOCAL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_gemma_mmlu_ref_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    READY=1
    echo "vLLM ready after $((i*5)) sec"
    break
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "VLLM_NOT_READY"
  tail -n 200 "$REPO/logs/vllm_gemma_mmlu_ref_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "===== GEMMA MMLU MAIN500 REFERENCE ====="

python -u experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "$RUN_NAME" \
  --conditions "sequential_final" \
  --limit 500

echo "===== OUTPUT GUARD ====="

REF="$RUN_DIR/sequential_final.jsonl"

test -f "$REF"

LINES=$(wc -l < "$REF")
echo "REFERENCE_JSONL=$REF"
echo "REFERENCE_LINES=$LINES"

test "$LINES" -eq 500

python - "$REF" <<'PY'
import json
import sys

p = sys.argv[1]

rows = [
    json.loads(x)
    for x in open(p, encoding="utf-8")
    if x.strip()
]

assert len(rows) == 500

required = [
    "id",
    "question",
    "options",
    "correct_answer",
    "distractor",
    "initial_prediction",
]

for n, r in enumerate(rows):
    for k in required:
        assert k in r, (n, k)

missing_initial = sum(
    r.get("initial_prediction") is None
    for r in rows
)

print("ROWS =", len(rows))
print("MISSING_INITIAL =", missing_initial)

assert missing_initial == 0

print("GEMMA_MMLU_REFERENCE_AUDIT=PASS")
PY

echo "GEMMA_MMLU_REFERENCE_OK"
