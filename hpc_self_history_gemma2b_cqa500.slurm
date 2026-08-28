#!/bin/bash
#SBATCH --job-name=selfhist_gemma_cqa
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_gemma_cqa_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_gemma_cqa_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/work/int/xuchengxue23/social_conformity_revision}"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="${ENV_PATH:-/gpfs/work/int/xuchengxue23/envs/self_history_vllm}"
MODEL_PATH="${MODEL_PATH:-$REPO/models/gemma2_2b_it}"
DATA_PATH_LOCAL="$PROJECT/data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json"
EXPECTED_MD5="cc5699972e9da35dd0fd2581b8bd2598"
MODEL_NAME_LOCAL="google/gemma-2-2b-it"
BASE_RUN_NAME="gemma2_2b_cqa_main500_selfhist_${SLURM_JOB_ID}"
CONTROL_RUN_NAME="gemma2_2b_cqa_main500_controls_${SLURM_JOB_ID}"
BASE_DIR="$PROJECT/results/runs/$BASE_RUN_NAME/2026-08-26_self_history_ablation"
CONTROL_DIR="$PROJECT/results/runs/$CONTROL_RUN_NAME/2026-08-26_self_history_ablation"
SELECTION_JSON="$PROJECT/results/runs/$BASE_RUN_NAME/neutral_control_selection.json"

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
test -f "$DATA_PATH_LOCAL"
DATA_MD5=$(md5sum "$DATA_PATH_LOCAL" | awk '{print $1}')
echo "DATA_MD5=$DATA_MD5"
test "$DATA_MD5" = "$EXPECTED_MD5"

cd "$PROJECT"
export PYTHONPATH=.
export MODEL_NAME="$MODEL_NAME_LOCAL"
export API_URL="http://127.0.0.1:8000/v1/chat/completions"
export DATA_PATH="$DATA_PATH_LOCAL"
export RESULT_DIR="results"
export USE_CONFIDENCE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

echo "===== STATIC PREFLIGHT ====="
python experiments/2026-08-26_self_history_ablation/test_prompt_equivalence.py

echo "===== START VLLM ====="
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME_LOCAL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_selfhist_gemma_cqa_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    READY=1
    echo "vLLM ready after $((i*5)) sec"
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  tail -n 150 "$REPO/logs/vllm_selfhist_gemma_cqa_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "===== STAGE 1: OBSERVED SELF-HISTORY CONDITIONS ====="
python experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "$BASE_RUN_NAME" \
  --conditions "sequential_final,stepwise_no_history,stepwise_answer_history,stepwise_answer_confidence_history" \
  --compute-tokens \
  --tokenizer-path "$MODEL_PATH"

echo "===== TOKENIZER-SPECIFIC NEUTRAL SELECTION ====="
python experiments/2026-08-26_self_history_ablation/select_neutral_controls.py \
  --tokenizer-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME_LOCAL" \
  --self-history-jsonl "$BASE_DIR/stepwise_answer_confidence_history.jsonl" \
  --output-json "$SELECTION_JSON"
python experiments/2026-08-26_self_history_ablation/verify_neutral_token_match.py \
  --tokenizer-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME_LOCAL" \
  --selection-json "$SELECTION_JSON"

echo "===== STAGE 2: MATCHED NEUTRAL CONTROLS ====="
python experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "$CONTROL_RUN_NAME" \
  --conditions "sequential_final_length_matched,sequential_final_neutral_v2,sequential_final_short_ack" \
  --reference-jsonl "$BASE_DIR/sequential_final.jsonl" \
  --neutral-selection-json "$SELECTION_JSON" \
  --compute-tokens \
  --tokenizer-path "$MODEL_PATH"

python experiments/2026-08-26_self_history_ablation/analyze_ablation.py \
  --run-dir "$BASE_DIR" \
  --condition "sequential_final_length_matched=$CONTROL_DIR/sequential_final_length_matched.jsonl" \
  --condition "sequential_final_neutral_v2=$CONTROL_DIR/sequential_final_neutral_v2.jsonl" \
  --condition "sequential_final_short_ack=$CONTROL_DIR/sequential_final_short_ack.jsonl" \
  --bootstrap-reps 10000 \
  --seed 12345 \
  --output-json "$PROJECT/results/runs/$CONTROL_RUN_NAME/self_history_analysis_combined.json"

echo "SELF_HISTORY_GEMMA_CQA_OK"
