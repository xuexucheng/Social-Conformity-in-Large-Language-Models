#!/bin/bash
#SBATCH --job-name=rat_q3_csqa
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/rationale_q3_csqa_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/rationale_q3_csqa_%j.err

set -euo pipefail

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="/gpfs/work/int/xuchengxue23/envs/self_history_vllm"

MODEL_PATH="$REPO/models/qwen25_3b_instruct"
MODEL_NAME_LOCAL="Qwen/Qwen2.5-3B-Instruct"

DATA_PATH_LOCAL="$PROJECT/data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json"
EXPECTED_MD5="94a7caf698a32bfbc3aed17443aebd80"

REFERENCE_JSONL="$PROJECT/results/runs/qwen25_3b_cqa_main500_2542766/2026-08-26_self_history_ablation/sequential_final.jsonl"

RUN_NAME="qwen25_3b_csqa_rationale_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT/results/runs/$RUN_NAME/2026-09-04_rationale_sensitivity"

PORT=$((18000 + SLURM_JOB_ID % 10000))

module purge
module load anaconda3/2022.10-gcc-8.5.0-4dp3trd
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "===== ENVIRONMENT ====="
hostname
date
nvidia-smi
python --version

test -d "$MODEL_PATH"
test -f "$DATA_PATH_LOCAL"
test -f "$REFERENCE_JSONL"
test "$(wc -l < "$REFERENCE_JSONL")" -eq 500

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

python - <<'PY'
import importlib.util
from pathlib import Path

p = Path(
    "experiments/2026-09-04_rationale_sensitivity/conditions.py"
)

spec = importlib.util.spec_from_file_location("C", p)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

assert C.N_WRONG_GUIDES == 5
assert hasattr(C, "build_sequential_final_rationale_messages")

print("RATIONALE_PREFLIGHT=PASS")
PY

echo "===== START VLLM ====="

python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME_LOCAL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_rationale_q3_csqa_${SLURM_JOB_ID}.log" 2>&1 &

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
    tail -n 200 "$REPO/logs/vllm_rationale_q3_csqa_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "===== RUN RATIONALE SENSITIVITY ====="

python -u \
  experiments/2026-09-04_rationale_sensitivity/run_rationale.py \
  --run-name "$RUN_NAME" \
  --conditions "sequential_final_rationale" \
  --limit 500 \
  --reference-jsonl "$REFERENCE_JSONL" \
  --compute-tokens \
  --tokenizer-path "$MODEL_PATH"

OUT="$RUN_DIR/sequential_final_rationale.jsonl"

echo "===== OUTPUT AUDIT ====="

test -f "$OUT"
test "$(wc -l < "$OUT")" -eq 500

python - "$OUT" <<'PY'
import json
import sys

path = sys.argv[1]

rows = [
    json.loads(x)
    for x in open(path, encoding="utf-8")
    if x.strip()
]

assert len(rows) == 500

for r in rows:
    assert r["condition"] == "sequential_final_rationale"
    assert len(r["agent_opinions"]) == 5
    assert r["agent_opinions"] == [r["distractor"]] * 5
    assert r["initial_answer_source"] == "reference_jsonl"

    msgs = r["final_messages"]

    rationale_lines = [
        m["content"]
        for m in msgs
        if m.get("role") == "user"
        and m.get("content", "").startswith("Agent ")
        and " suggests that option " in m.get("content", "")
    ]

    assert len(rationale_lines) == 5

print("ROWS =", len(rows))
print("RATIONALE_OUTPUT_AUDIT=PASS")
PY

echo "OUTPUT_JSONL=$OUT"
echo "RATIONALE_QWEN3B_CSQA_OK"
