#!/usr/bin/env bash
set -e

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096