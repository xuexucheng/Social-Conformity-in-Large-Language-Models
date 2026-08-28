#!/bin/bash
#SBATCH --job-name=selfhist_q3b_ctrl
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_q3b_controls_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_q3b_controls_%j.err

set -euo pipefail

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
ENV="/gpfs/work/int/xuchengxue23/envs/self_history_vllm"
MODEL="$REPO/models/qwen25_3b_instruct"
DATA="/gpfs/work/int/xuchengxue23/anti_conformity_llm/data/old_conformity/commonsenseqa_five_wrong_guidance_refined_500.json"
BASE_RUN="$PROJECT/results/runs/qwen25_3b_cqa_main500_2542766/2026-08-26_self_history_ablation"
REFERENCE="$BASE_RUN/sequential_final.jsonl"
RUN_NAME="qwen25_3b_cqa_neutral_controls_${SLURM_JOB_ID}"
NEW_RUN="$PROJECT/results/runs/$RUN_NAME/2026-08-26_self_history_ablation"

cd "$REPO"

module purge
module load anaconda3/2022.10-gcc-8.5.0-4dp3trd
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "===== BASIC INFO ====="
echo "node=$(hostname)"
date
nvidia-smi

echo "===== GIT ====="
git branch --show-current
git log -1 --oneline

echo "===== INPUT GUARDS ====="
MD5=$(md5sum "$DATA" | awk '{print $1}')
echo "DATA_MD5=$MD5"
if [ "$MD5" != "cc5699972e9da35dd0fd2581b8bd2598" ]; then
    echo "ERROR: wrong main500 dataset"
    exit 1
fi
if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: missing initial-answer reference: $REFERENCE"
    exit 1
fi
if [ "$(wc -l < "$REFERENCE")" -ne 500 ]; then
    echo "ERROR: reference JSONL does not contain 500 lines"
    exit 1
fi

cd "$PROJECT"

export PYTHONPATH=.
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
export API_URL="http://127.0.0.1:8000/v1/chat/completions"
export DATA_PATH="$DATA"
export RESULT_DIR="results"

echo "===== TOKEN PREFLIGHT ====="
python experiments/2026-08-26_self_history_ablation/verify_neutral_token_match.py \
  --tokenizer-path "$MODEL" \
  --model-name "$MODEL_NAME"

echo "===== START VLLM ====="
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "Qwen/Qwen2.5-3B-Instruct" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_selfhist_q3b_controls_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "===== WAIT FOR VLLM ====="
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
    echo "ERROR: vLLM failed to start"
    tail -n 150 "$REPO/logs/vllm_selfhist_q3b_controls_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "===== RUN TWO ADD-ON CONTROLS ====="
python experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "$RUN_NAME" \
  --conditions "sequential_final_neutral_v2,sequential_final_short_ack" \
  --reference-jsonl "$REFERENCE" \
  --compute-tokens \
  --tokenizer-path "$MODEL"

echo "===== COMBINED ANALYSIS ====="
python experiments/2026-08-26_self_history_ablation/analyze_ablation.py \
  --run-dir "$BASE_RUN" \
  --condition "sequential_final_neutral_v2=$NEW_RUN/sequential_final_neutral_v2.jsonl" \
  --condition "sequential_final_short_ack=$NEW_RUN/sequential_final_short_ack.jsonl" \
  --bootstrap-reps 10000 \
  --seed 12345 \
  --output-json "$PROJECT/results/runs/$RUN_NAME/self_history_analysis_combined.json"

echo "===== OUTPUT FILES ====="
find "$PROJECT/results/runs/$RUN_NAME" -maxdepth 4 -type f -print
echo "NEUTRAL_CONTROLS_OK"
