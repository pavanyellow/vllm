#!/usr/bin/env bash
# Runtime env vars for qwen-3.6-a3b-a100. Source this before start_server.sh
# or any python script that imports vllm.
#
# Inputs (set by setup.sh or override at deploy time):
#   APP_DIR        default: /app
#   VENV_DIR       default: $APP_DIR/venv

APP_DIR="${APP_DIR:-/app}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"

# Activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# nvidia libs ship inside the venv site-packages
NVIDIA="$VENV_DIR/lib/python3.12/site-packages/nvidia"
TORCH_LIB="$VENV_DIR/lib/python3.12/site-packages/torch/lib"
CU13="$NVIDIA/cu13"

export LD_LIBRARY_PATH="$CU13/lib:$NVIDIA/cublas/lib:$NVIDIA/cudnn/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cuda_nvrtc/lib:$NVIDIA/cufft/lib:$NVIDIA/curand/lib:$NVIDIA/cusolver/lib:$NVIDIA/cusparse/lib:$NVIDIA/cusparselt/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$NVIDIA/nvtx/lib:$NVIDIA/nvshmem/lib:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# DCGM proxy can shadow libcublas symbols at dlopen — LD_PRELOAD the venv copies.
if [ -f "$NVIDIA/cublas/lib/libcublas.so.12" ]; then
  export LD_PRELOAD="$NVIDIA/cublas/lib/libcublasLt.so.12:$NVIDIA/cublas/lib/libcublas.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# nvcc / cuda toolkit (some kernels JIT at runtime)
export CUDA_HOME="$CU13"
export PATH="$CU13/bin:$PATH"

# vLLM tuning
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_V1=1
# flashinfer's bundled cccl headers conflict with our nvcc 13 — disable its sampler
# (PyTorch-native fallback; minor decode impact, no TTFT effect)
export VLLM_USE_FLASHINFER_SAMPLER=0
# Reduce alloc fragmentation when adding extended cudagraph captures
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Better HF download throughput
export HF_HUB_ENABLE_HF_TRANSFER=1
