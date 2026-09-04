#!/bin/bash
#SBATCH --job-name=peer_qwen_csqa
#SBATCH --partition=gpu4090
#SBATCH --qos=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/peer_qwen_csqa_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/logs/peer_qwen_csqa_%j.err

set -euo pipefail

: "${PEER_COUNT:?set PEER_COUNT=1 or 3}"
: "${REFERENCE_JSONL:?set REFERENCE_JSONL to canonical Qwen3B-CSQA reference}"

if [ "$PEER_COUNT" != "1" ] && [ "$PEER_COUNT" != "3" ]; then
    echo "ERROR: new jobs are only for PEER_COUNT=1 or 3"
    exit 1
fi

REPO="/gpfs/work/int/xuchengxue23/social_conformity_revision"
PROJECT="$REPO/agent_conformity_project"
ENV_PATH="/gpfs/work/int/xuchengxue23/envs/self_history_vllm"

MODEL_PATH="$REPO/models/qwen25_3b_instruct"
MODEL_NAME_LOCAL="Qwen/Qwen2.5-3B-Instruct"

DATA_PATH_LOCAL="$PROJECT/data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json"
EXPECTED_MD5="94a7caf698a32bfbc3aed17443aebd80"

RUN_NAME="qwen25_3b_csqa_peercount${PEER_COUNT}_${SLURM_JOB_ID}"
RUN_DIR="$PROJECT/results/runs/$RUN_NAME/2026-09-03_peer_count_sensitivity"

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
echo "PEER_COUNT=$PEER_COUNT"
echo "REFERENCE_JSONL=$REFERENCE_JSONL"

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
export PEER_COUNT="$PEER_COUNT"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "===== PEER-COUNT STATIC PREFLIGHT ====="

python - <<'PY'
import importlib.util
import os
from pathlib import Path

p = Path(
    "experiments/2026-09-03_peer_count_sensitivity/conditions.py"
)

spec = importlib.util.spec_from_file_location(
    "peer_conditions", p
)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

expected = int(os.environ["PEER_COUNT"])

print("N_WRONG_GUIDES =", C.N_WRONG_GUIDES)
print(
    "PEER_ORDER =",
    list(range(1, C.N_WRONG_GUIDES + 1))
)

assert C.N_WRONG_GUIDES == expected
assert expected in {1, 3}

print("PEER_COUNT_PREFLIGHT=PASS")
PY

echo "===== START VLLM ====="

python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME_LOCAL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > "$REPO/logs/vllm_peer_qwen_csqa_${PEER_COUNT}_${SLURM_JOB_ID}.log" 2>&1 &

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
    tail -n 200 \
      "$REPO/logs/vllm_peer_qwen_csqa_${PEER_COUNT}_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "===== RUN PEER-COUNT SENSITIVITY ====="

python -u \
  experiments/2026-09-03_peer_count_sensitivity/run_peer_count.py \
  --run-name "$RUN_NAME" \
  --conditions "sequential_final" \
  --limit 500 \
  --reference-jsonl "$REFERENCE_JSONL" \
  --compute-tokens \
  --tokenizer-path "$MODEL_PATH"

OUT="$RUN_DIR/sequential_final.jsonl"

echo "===== OUTPUT AUDIT ====="

test -f "$OUT"
test "$(wc -l < "$OUT")" -eq 500

python - "$OUT" "$PEER_COUNT" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

rows = [
    json.loads(x)
    for x in open(path, encoding="utf-8")
    if x.strip()
]

assert len(rows) == 500

for r in rows:
    assert len(r["agent_opinions"]) == expected
    assert len(r["peer_order"]) == expected

    assert r["agent_opinions"] == [
        r["distractor"]
    ] * expected

    assert r.get("initial_answer_source") == "reference_jsonl"

print("ROWS =", len(rows))
print("PEER_COUNT =", expected)
print("PEER_SIGNAL_AUDIT=PASS")
PY

echo "OUTPUT_JSONL=$OUT"
echo "PEER_COUNT_QWEN_CSQA_OK"
