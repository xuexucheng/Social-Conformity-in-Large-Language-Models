#!/bin/bash
#SBATCH --job-name=clean_label
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/clean_label_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/clean_label_%j.err

set -euo pipefail

# Required submit-time variables:
#   MODEL_PATH, MODEL_NAME_LOCAL, REFERENCE_JSONL, RUN_TAG
# Example:
# sbatch --export=ALL,MODEL_PATH=/path/to/model,MODEL_NAME_LOCAL=Qwen/Qwen2.5-3B-Instruct,REFERENCE_JSONL=/path/to/sequential_final.jsonl,RUN_TAG=qwen3b hpc_clean_social_label.slurm

: "${MODEL_PATH:?set MODEL_PATH to the exact local checkpoint}"
: "${MODEL_NAME_LOCAL:?set MODEL_NAME_LOCAL to the served model ID}"
: "${REFERENCE_JSONL:?set REFERENCE_JSONL to an archived 500-row initial-answer source}"
: "${RUN_TAG:?set RUN_TAG to a short model tag}"

REPO="${REPO:-/gpfs/work/int/xuchengxue23/social_conformity_revision}"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="${ENV_PATH:-/gpfs/work/int/xuchengxue23/envs/self_history_vllm}"
DATA_PATH_LOCAL="$PROJECT/data/datasets/mmlu_all_validation.json"
EXPECTED_MD5="598fec26af1127bfd7bca06c39c1a159"
RUN_NAME="clean_social_label_${RUN_TAG}_${SLURM_JOB_ID}"
SELECTION_JSON="$PROJECT/results/runs/$RUN_NAME/neutral_label_selection.json"

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
test -d "$MODEL_PATH"
test -f "$REFERENCE_JSONL"
test "$(wc -l < "$REFERENCE_JSONL")" -eq 500
DATA_MD5=$(md5sum "$DATA_PATH_LOCAL" | awk '{print $1}')
echo "DATA_MD5=$DATA_MD5"
test "$DATA_MD5" = "$EXPECTED_MD5"

cd "$PROJECT"
export PYTHONPATH=.
export MODEL_NAME="$MODEL_NAME_LOCAL"
PORT=$((18000 + SLURM_JOB_ID % 10000))
export API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
echo "VLLM_PORT=$PORT"
export DATA_PATH="$DATA_PATH_LOCAL"
export RESULT_DIR="results"
export USE_CONFIDENCE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

echo "===== OFFLINE LABEL-ONLY PREFLIGHT ====="
python experiments/2026-08-29_clean_social_label/test_conditions.py
python experiments/2026-08-29_clean_social_label/select_neutral_label.py \
  --tokenizer-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME_LOCAL" \
  --data-path "$DATA_PATH_LOCAL" \
  --limit 500 \
  --output-json "$SELECTION_JSON"

echo "===== START VLLM ====="
PYTHONUNBUFFERED=1 python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME_LOCAL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_clean_label_${RUN_TAG}_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null; then
    READY=1
    echo "vLLM ready after $((i*5)) sec"
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  tail -n 150 "$REPO/logs/vllm_clean_label_${RUN_TAG}_${SLURM_JOB_ID}.log"
  exit 1
fi

python experiments/2026-08-29_clean_social_label/run_experiment.py \
  --run-name "$RUN_NAME" \
  --limit 500 \
  --reference-jsonl "$REFERENCE_JSONL" \
  --label-selection-json "$SELECTION_JSON" \
  --compute-tokens \
  --tokenizer-path "$MODEL_PATH"

echo "CLEAN_SOCIAL_LABEL_OK"
