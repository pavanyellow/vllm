# qwen-3.6-a3b-a100 — voice-agent INT8 W8A8 vLLM server
#
# Image layout inside container:
#   /app          ← this folder (scripts, env.sh, config/)
#   /app/venv     ← Python 3.12 + torch 2.11 cu13 + vllm 0.22 + patches
#   /data/models  ← model weights (MOUNT a persistent volume here on first run;
#                   first start auto-downloads ~33 GB if empty)
#
# Build:    docker build -t qwen-a100:latest -f Dockerfile .
# Run:      docker run --gpus all --rm -it -p 8000:8000 \
#               -v qwen-models:/data/models -e HF_TOKEN=... qwen-a100:latest
#
# Health:   curl http://localhost:8000/health
# Metrics:  curl http://localhost:8000/metrics

# Base: NVIDIA CUDA 12.x runtime on Ubuntu 22.04 — works on A100/H100/L40/etc.
# (vLLM 0.22 + torch 2.11 ship cu13 nvidia-* python packages so the BASE
# CUDA version mostly matters for the driver/runtime compat; vllm brings its own.)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_DIR=/app
ENV VENV_DIR=/app/venv
ENV MODEL_DIR=/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8
ENV PYTHONUNBUFFERED=1

# ─── system packages ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      gcc g++ make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ─── app folder + venv + python deps ──────────────────────────────────────
WORKDIR /app
# Copy code FIRST so setup.sh + apply_vllm_patches.sh + bundled config are present.
COPY . /app/

# Build venv and install (heavy; takes 5-10 min, ~6 GB image layer)
RUN python3.12 -m venv "$VENV_DIR" \
 && . "$VENV_DIR/bin/activate" \
 && pip install -U pip setuptools wheel \
 && pip install \
      'vllm==0.22.0' 'transformers>=4.57.1' 'huggingface_hub' 'hf_transfer' \
      'numpy' 'safetensors' 'tokenizers' 'openai>=1.40' 'fastapi' 'uvicorn'

# ─── apply vLLM patches + nvrtc symlinks + bundled tuned MoE config ─────
RUN . "$VENV_DIR/bin/activate" && bash "$APP_DIR/apply_vllm_patches.sh"

# ─── runtime ─────────────────────────────────────────────────────────────
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Entrypoint: first start downloads model if needed (one-time, ~5-10 min on a
# good network) AND auto-runs autotune if no device-named config exists, then
# launches the vllm OpenAI server.
ENTRYPOINT ["bash", "-lc", "/app/setup.sh && /app/start_server.sh"]
