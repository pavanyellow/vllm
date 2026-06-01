#!/usr/bin/env bash
# qwen-3.6-a3b-a100 — one-shot installer for bare-metal RunPod pods.
# Idempotent: re-run is safe (skips already-done steps).
#
# Inputs (env vars, all optional):
#   APP_DIR        where the app lives. Default: /app
#   MODEL_DIR      where to put model weights. Default: /data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8
#   VENV_DIR       where to create the Python venv. Default: $APP_DIR/venv
#   HF_TOKEN       HuggingFace token (recommended; anon downloads are rate-limited)
#   SKIP_MODEL=1   skip model download (assume already at MODEL_DIR)
#   SKIP_AUTOTUNE=1 skip auto-tuning even if no device config exists
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " qwen-3.6-a3b-a100 setup"
echo "=========================================="
echo "  APP_DIR=$APP_DIR"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  VENV_DIR=$VENV_DIR"
echo "  SELF_DIR=$SELF_DIR"
echo "  HF_TOKEN=${HF_TOKEN:+set}${HF_TOKEN:-not set}"
echo

# ──────────────────────────────────────────────────────────────────────────
# 1. System packages: Python 3.12, gcc, headers (needed for Triton JIT)
# ──────────────────────────────────────────────────────────────────────────
echo "[1/7] system packages"
if ! command -v python3.12 >/dev/null; then
  if command -v apt-get >/dev/null; then
    # Ubuntu / Debian (most RunPod base images)
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.12 python3.12-venv python3.12-dev gcc git curl
  elif command -v dnf >/dev/null; then
    # Rocky / Alma / CentOS Stream
    dnf install -y python3.12 python3.12-devel gcc git curl
  else
    echo "ERROR: no apt-get or dnf; install python3.12 + gcc + git + curl manually"
    exit 1
  fi
fi
python3.12 --version

# ──────────────────────────────────────────────────────────────────────────
# 2. Python venv + base tools
# ──────────────────────────────────────────────────────────────────────────
echo "[2/7] venv at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  python3.12 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -U pip setuptools wheel

# ──────────────────────────────────────────────────────────────────────────
# 3. Install vLLM 0.22 + torch 2.11.0+cu130 + supporting libs
#    NOTE: vllm pulls in torch with the right cu13 nvidia-* packages.
#    No manual --index-url juggling needed in vLLM 0.22.
# ──────────────────────────────────────────────────────────────────────────
echo "[3/7] vllm 0.22 + transformers + hf_transfer"
pip install \
  'vllm==0.22.0' \
  'transformers>=4.57.1' \
  'huggingface_hub' \
  'hf_transfer' \
  'numpy' \
  'safetensors' \
  'tokenizers' \
  'openai>=1.40' \
  'fastapi' \
  'uvicorn'

python -c "import torch, vllm; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()} vllm={vllm.__version__}')"

# ──────────────────────────────────────────────────────────────────────────
# 4. Apply the 3 vLLM patches (idempotent)
# ──────────────────────────────────────────────────────────────────────────
echo "[4/7] apply vLLM patches + nvrtc symlink + bundled tuned MoE config"
bash "$SELF_DIR/apply_vllm_patches.sh"

# ──────────────────────────────────────────────────────────────────────────
# 5. Download model (~33.5 GiB, ~5-10 min on a fast network)
# ──────────────────────────────────────────────────────────────────────────
echo "[5/7] model download"
if [ "${SKIP_MODEL:-0}" = "1" ]; then
  echo "  SKIP_MODEL=1, skipping"
elif [ -f "$MODEL_DIR/config.json" ]; then
  echo "  $MODEL_DIR/config.json exists, skipping download"
else
  mkdir -p "$(dirname "$MODEL_DIR")"
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  ${HF_TOKEN:+HF_TOKEN=$HF_TOKEN} \
  python <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8",
    local_dir="$MODEL_DIR",
    max_workers=8,
    token=os.environ.get("HF_TOKEN"),
)
print("model download complete")
PY
fi

# ──────────────────────────────────────────────────────────────────────────
# 6. Fix the broken tokenizer_config.json (Quark export bug)
#    Replace with the Qwen2Tokenizer config from the BF16 base model.
# ──────────────────────────────────────────────────────────────────────────
echo "[6/7] tokenizer fix"
if grep -q '"TokenizersBackend"' "$MODEL_DIR/tokenizer_config.json"; then
  cp "$MODEL_DIR/tokenizer_config.json" "$MODEL_DIR/tokenizer_config.json.broken"
  python <<PY
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="Qwen/Qwen3.6-35B-A3B",
    filename="tokenizer_config.json",
    local_dir="/tmp/qwen-bf16-tok",
    token=os.environ.get("HF_TOKEN"),
)
import shutil
shutil.copy(p, "$MODEL_DIR/tokenizer_config.json")
print("tokenizer_config.json replaced with Qwen2Tokenizer version")
PY
else
  echo "  tokenizer_config.json already has Qwen2Tokenizer (skipping)"
fi

# ──────────────────────────────────────────────────────────────────────────
# 7. Autotune MoE kernel if no device config exists (one-time, ~30 min)
# ──────────────────────────────────────────────────────────────────────────
echo "[7/7] device-specific MoE config"
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '_'))")
CFG_FILE_NAME="E=256,N=512,device_name=${GPU_NAME}.json"
VLLM_CFG_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/model_executor/layers/fused_moe/configs
if [ -f "$VLLM_CFG_DIR/$CFG_FILE_NAME" ]; then
  echo "  device config already in place: $CFG_FILE_NAME"
elif [ "${SKIP_AUTOTUNE:-0}" = "1" ]; then
  echo "  SKIP_AUTOTUNE=1, skipping (vLLM will use heuristic default — suboptimal)"
else
  echo "  no config for $GPU_NAME — running autotune (~30 min one-time)"
  source "$SELF_DIR/env.sh"
  python "$SELF_DIR/autotune_moe.py" \
    --num-experts 256 --top-k 8 --hidden-size 2048 \
    --shard-intermediate-size 512 --dtype int8_w8a8 \
    --token-counts 4096
  # mirror to the no-dtype-suffix name (until vLLM patch #2 merges upstream)
  cp "$VLLM_CFG_DIR/E=256,N=512,device_name=${GPU_NAME},dtype=int8_w8a8.json" \
     "$VLLM_CFG_DIR/$CFG_FILE_NAME"
  echo "  autotune complete, config saved"
fi

echo
echo "=========================================="
echo " setup DONE. Next:"
echo "   source $VENV_DIR/bin/activate"
echo "   source $SELF_DIR/env.sh"
echo "   bash $SELF_DIR/start_server.sh"
echo "=========================================="
