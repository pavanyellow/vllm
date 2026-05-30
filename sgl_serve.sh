#!/usr/bin/env bash
# Launch SGLang server for Qwen3.5-MoE on H100 SXM from the clean venv.
# Mirrors the vLLM E8 setup: ctx 32768, mem-frac 0.90, chunked-prefill 8192.
cd /tmp
export HF_HOME=/workspace/.hf
unset PYTHONPATH
exec /vllm-workspace/sgl-venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 8192 \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 --port 30000
