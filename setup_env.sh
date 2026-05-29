#!/usr/bin/env bash
#
# setup_env.sh — create the Python venv and install vLLM from this checkout.
#
# Installs the precompiled CUDA wheel (VLLM_USE_PRECOMPILED=1), so Python-only
# changes need no local C++/CUDA build. After this runs, start the server with:
#
#     source .venv/bin/activate     # puts .venv/bin (ninja) on PATH for JIT
#     export CUDA_HOME=/usr/local/cuda
#     vllm serve Qwen/Qwen3.6-35B-A3B-FP8 ...   # see README.md
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

command -v uv >/dev/null 2>&1 || {
    echo "error: 'uv' is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
}

# 1. Create the venv if it doesn't already exist.
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo ">> Creating venv (Python ${PYTHON_VERSION}) at ${VENV_DIR}"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    echo ">> Reusing existing venv at ${VENV_DIR}"
fi

# 2. Install vLLM (editable) from this checkout using the precompiled wheel.
#    `ninja` comes in as a dependency here — it's needed at runtime for the
#    FlashInfer / GDN linear-attention JIT kernels.
echo ">> Installing vLLM (VLLM_USE_PRECOMPILED=1) ..."
VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="${VENV_DIR}" \
    uv pip install -e "${REPO_DIR}" --torch-backend=auto

# 3. Sanity check.
echo ">> Verifying import ..."
"${VENV_DIR}/bin/python" - <<'PY'
import torch, vllm
print(f"vllm {vllm.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
PY

cat <<EOF

Done. To launch the server:

    source ${VENV_DIR}/bin/activate
    export CUDA_HOME=/usr/local/cuda
    # then run the 'vllm serve ...' command from README.md
EOF
