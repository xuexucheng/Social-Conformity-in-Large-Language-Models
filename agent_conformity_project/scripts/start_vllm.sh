#!/usr/bin/env bash
set -e

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# Paper models:
#   MODEL_NAME=Qwen/Qwen2.5-3B-Instruct bash scripts/start_vllm.sh
#   MODEL_NAME=microsoft/Phi-3.5-mini-instruct bash scripts/start_vllm.sh
#   MODEL_NAME=google/gemma-2-2b-it bash scripts/start_vllm.sh
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
