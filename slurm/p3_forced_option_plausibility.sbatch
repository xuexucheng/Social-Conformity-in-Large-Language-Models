#!/bin/bash
#SBATCH --job-name=p3-plausibility
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/slurm/logs/p3_%A_%a.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/slurm/logs/p3_%A_%a.err

set -euo pipefail

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
PY="/gpfs/work/int/xuchengxue23/envs/self_history_vllm/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS=(qwen phi gemma)
MODEL_KEY="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "P3 FORCED-OPTION PLAUSIBILITY"
echo "job_id=$SLURM_JOB_ID"
echo "array_task=$SLURM_ARRAY_TASK_ID"
echo "model_key=$MODEL_KEY"
echo "host=$(hostname)"
echo "start=$(date -Is)"
echo "========================================"

nvidia-smi

"$PY" analysis/run_forced_option_model_pair.py \
    --model-key "$MODEL_KEY" \
    --output-dir analysis/forced_option_plausibility \
    --resume

echo "========================================"
echo "finish=$(date -Is)"
echo "P3 TASK COMPLETE: $MODEL_KEY"
echo "========================================"
