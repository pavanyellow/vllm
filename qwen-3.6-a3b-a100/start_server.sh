#!/usr/bin/env bash
# Start the OpenAI-compatible vLLM server with all the tuned settings.
# Endpoints (default port 8000):
#   POST /v1/chat/completions    OpenAI chat API (supports stream=true)
#   POST /v1/completions         OpenAI completions API
#   GET  /v1/models              list served models
#   GET  /health                 200 when ready
#   GET  /metrics                Prometheus-format vLLM metrics
#
# Inputs (env vars):
#   MODEL_DIR       default: /data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8
#   PORT            default: 8000
#   API_KEY         default: unset (no auth). Set sk-... to require auth.
#   MAX_MODEL_LEN   default: 6300 (covers L=4K + headroom)
#   MAX_NUM_SEQS    default: 4 (single-stream voice agent)
#   GPU_MEM_UTIL    default: 0.85
#   EXTENDED_CG     default: 1 (add prefill-sized captures [640..2048 stride 128])
#                   set EXTENDED_CG=0 to use vLLM defaults only (no prefill captures)
#   WARMUP          default: 1 (send 3 synthetic warmup requests after server ready)
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF_DIR/env.sh"

MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6300}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
EXTENDED_CG="${EXTENDED_CG:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-3.6-a3b}"

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: model not found at $MODEL_DIR. Run setup.sh first."; exit 1; }

# Build the compilation_config JSON that pins our extended cudagraph capture sizes.
# Default capture sizes are [1, 2, 4] + range(8, 256, 8) + range(256, 513, 16).
# We extend with prefill-sized captures every 128 from 640 to 2048.
if [ "$EXTENDED_CG" = "1" ]; then
  COMPILATION_CFG=$(python <<'PY'
import json
default_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
extras = [640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
sizes = sorted(set(default_sizes + extras))
print(json.dumps({
    "cudagraph_capture_sizes": sizes,
    "max_cudagraph_capture_size": sizes[-1],
}))
PY
)
  CG_FLAGS=(--compilation-config "$COMPILATION_CFG")
else
  CG_FLAGS=()
fi

API_KEY_FLAG=()
[ -n "${API_KEY:-}" ] && API_KEY_FLAG=(--api-key "$API_KEY")

echo "=========================================="
echo " starting vllm serve"
echo "   model=$MODEL_DIR"
echo "   served-as=$SERVED_MODEL_NAME"
echo "   port=$PORT"
echo "   max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM_UTIL"
echo "   extended_cg=$EXTENDED_CG"
echo "   api_key=${API_KEY:+set}${API_KEY:-NOT SET (open)}"
echo "=========================================="

# Run vllm serve in the foreground. Logs go to stdout (container will capture).
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size 1 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --trust-remote-code \
  --dtype auto \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  "${CG_FLAGS[@]}" \
  "${API_KEY_FLAG[@]}"
