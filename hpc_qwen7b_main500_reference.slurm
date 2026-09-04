#!/bin/bash
#SBATCH --job-name=qwen7b_ref
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/qwen7b_ref_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/qwen7b_ref_%j.err

set -euo pipefail

: "${DATA_KIND:?set DATA_KIND=csqa or mmlu}"

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="/gpfs/work/int/xuchengxue23/envs/self_history_vllm"

MODEL_PATH="$REPO/models/qwen25_7b_instruct"
MODEL_NAME_LOCAL="Qwen/Qwen2.5-7B-Instruct"

case "$DATA_KIND" in
  csqa)
    DATA_PATH_LOCAL="$PROJECT/data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json"
    EXPECTED_MD5="94a7caf698a32bfbc3aed17443aebd80"
    ;;
  mmlu)
    DATA_PATH_LOCAL="$PROJECT/data/datasets/mmlu_all_validation.json"
    EXPECTED_MD5="598fec26af1127bfd7bca06c39c1a159"
    ;;
  *)
    echo "ERROR: DATA_KIND must be csqa or mmlu"
    exit 1
    ;;
esac

RUN_NAME="qwen25_7b_${DATA_KIND}_main500_reference_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT/results/runs/$RUN_NAME/2026-08-26_self_history_ablation"

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

echo "===== INPUT GUARDS ====="
echo "DATA_KIND=$DATA_KIND"

test -x "$ENV_PATH/bin/python"
test -d "$MODEL_PATH"
test -f "$MODEL_PATH/config.json"
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
  > "$REPO/logs/vllm_qwen7b_${DATA_KIND}_${SLURM_JOB_ID}.log" 2>&1 &

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
  tail -n 200 "$REPO/logs/vllm_qwen7b_${DATA_KIND}_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "===== QWEN7B ${DATA_KIND} MAIN500 REFERENCE ====="

python -u experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "$RUN_NAME" \
  --conditions "sequential_final" \
  --limit 500

REF="$RUN_DIR/sequential_final.jsonl"

echo "===== OUTPUT AUDIT ====="
test -f "$REF"

python - "$REF" <<'PY'
import json
import sys

p = sys.argv[1]

rows = [
    json.loads(x)
    for x in open(p, encoding="utf-8")
    if x.strip()
]

print("REFERENCE_ROWS =", len(rows))

assert len(rows) == 500

missing = [
    r for r in rows
    if r.get("initial_prediction") is None
]

print("MISSING_INITIAL =", len(missing))

if missing:
    for r in missing[:10]:
        print(
            "BAD_INITIAL",
            r.get("id"),
            repr(r.get("raw_initial_output"))
        )
    raise SystemExit(2)

required = [
    "id",
    "question",
    "options",
    "correct_answer",
    "distractor",
    "initial_prediction",
]

for n, r in enumerate(rows):
    for field in required:
        assert field in r, (n, field)

print("QWEN7B_REFERENCE_AUDIT=PASS")
PY

echo "REFERENCE_JSONL=$REF"
echo "QWEN7B_REFERENCE_OK"
