#!/bin/bash
#SBATCH --job-name=selfhist_q3b_500
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_q3b_main500_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/selfhist_q3b_main500_%j.err

set -euo pipefail

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
ENV="/gpfs/work/int/xuchengxue23/envs/self_history_vllm"
MODEL="$REPO/models/qwen25_3b_instruct"
DATA="/gpfs/work/int/xuchengxue23/anti_conformity_llm/data/old_conformity/commonsenseqa_five_wrong_guidance_refined_500.json"

cd "$REPO"

module purge
module load anaconda3/2022.10-gcc-8.5.0-4dp3trd
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "===== BASIC INFO ====="
echo "node=$(hostname)"
date
nvidia-smi

echo "===== ENV ====="
which python
python --version
python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available())"
python -c "import vllm; print('vllm=',vllm.__version__)"
python -c "import transformers; print('transformers=',transformers.__version__)"

echo "===== GIT ====="
git branch --show-current
git log -1 --oneline

echo "===== DATA ====="
MD5=$(md5sum "$DATA" | awk '{print $1}')
echo "DATA_MD5=$MD5"

if [ "$MD5" != "cc5699972e9da35dd0fd2581b8bd2598" ]; then
    echo "ERROR: wrong main500 dataset"
    exit 1
fi

echo "===== MODEL ====="
du -sh "$MODEL"
ls -lh "$MODEL"/model-*.safetensors

cd "$PROJECT"

export PYTHONPATH=.
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
export API_URL="http://127.0.0.1:8000/v1/chat/completions"
export DATA_PATH="$DATA"
export RESULT_DIR="results"
export COMPUTE_TOKENS=1
export TOKENIZER_PATH="$MODEL"

echo "===== START VLLM ====="

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "Qwen/Qwen2.5-3B-Instruct" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_selfhist_q3b_${SLURM_JOB_ID}.log" 2>&1 &

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
    tail -n 150 "$REPO/logs/vllm_selfhist_q3b_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "===== MODEL ENDPOINT ====="
curl -s http://127.0.0.1:8000/v1/models
echo

echo "===== RUN FORMAL MAIN500 ====="

python experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name "qwen25_3b_cqa_main500_${SLURM_JOB_ID}" \
  --compute-tokens \
  --tokenizer-path "$MODEL"

echo "===== OUTPUT FILES ====="

find \
  "results/runs/qwen25_3b_cqa_main500_${SLURM_JOB_ID}" \
  -maxdepth 4 \
  -type f \
  -print

echo "===== MAIN500 FINISHED ====="
echo "MAIN500_OK"
