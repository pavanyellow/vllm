#!/usr/bin/env bash
# Clean SGLang 0.5.12.post1 install for Qwen3.5-MoE (qwen3_5) on H100 SXM, CUDA 13 / torch 2.11+cu130.
# venv is isolated (no system vllm). torch already installed from cu130 index.
# We install sglang --no-deps + its real runtime closure, because the published
# sglang wheel hard-pins flash-attn-4>=4.0.0b9 (no wheel -> source build, fails)
# and several GPU-kernel pkgs (tilelang/quack/deep-gemm/cutlass-dsl) the qwen3_5
# forward path does not use. Kernel is the cu130 build pinned to sglang's ==0.4.2.post2.
set -e
P=/vllm-workspace/sgl-venv/bin/python

echo "### 1. cu130 sgl-kernel (exact pin 0.4.2.post2) ###"
$P -m pip install -q /tmp/skdl/sglang_kernel-0.4.2.post2+cu130-cp310-abi3-manylinux2014_x86_64.whl

echo "### 2. flashinfer (JIT wheels from PyPI, sglang's pinned post1) ###"
$P -m pip install -q "flashinfer_python==0.6.11.post1" "flashinfer_cubin==0.6.11.post1"

echo "### 3. runtime closure (sglang deps minus flash-attn-4 + exotic kernels) ###"
$P -m pip install -q \
  "transformers==5.6.0" "xgrammar==0.2.0" "torchao==0.17.0" \
  "outlines==0.1.11" "openai==2.6.1" "blobfile==3.0.0" "timm==1.0.16" \
  "llguidance<0.8.0,>=0.7.11" "cuda-python>=13.0" "mistral_common>=1.11.0" \
  "torch_memory_saver>=0.0.9.post1" "pyzmq>=25.1.2" "prometheus-client>=0.20.0" \
  aiohttp anthropic compressed-tensors datasets einops fastapi gguf interegular \
  msgspec ninja easydict numpy nvidia-ml-py orjson packaging partial_json_parser \
  pillow psutil pybase64 pydantic python-multipart requests scipy sentencepiece \
  setproctitle "soundfile==0.13.1" tiktoken tqdm uvicorn uvloop watchfiles IPython \
  modelscope py-spy build hf_transfer

echo "### 4. sglang itself (no-deps) ###"
$P -m pip install -q --no-deps "sglang==0.5.12.post1"

echo "### DONE — versions ###"
$P -m pip list 2>/dev/null | grep -iE "^(sglang|sglang-kernel|flashinfer|torch|transformers|xgrammar) "
