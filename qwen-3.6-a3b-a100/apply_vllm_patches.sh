#!/usr/bin/env bash
# Apply the 3 vLLM 0.22 patches in-place to the installed package.
# Idempotent — re-running is safe (each patch checks if its fix is already present).
#
# These mirror upstream_pr/0001..0003. PR review takes time; we apply locally
# so deployments work today.
set -euo pipefail

VLLM_LIB="${VLLM_LIB:-$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
echo "[patches] target: $VLLM_LIB"
[ -d "$VLLM_LIB" ] || { echo "ERROR: VLLM_LIB not found"; exit 1; }

# ──────────────────────────────────────────────────────────────────────────
# Patch 1: QuarkW8A8Int8MoEMethod missing moe_quant_config assignment
# ──────────────────────────────────────────────────────────────────────────
QM="$VLLM_LIB/model_executor/layers/quantization/quark/quark_moe.py"
if grep -qE 'self\.moe_quant_config = self\.get_fused_moe_quant_config\(layer\)' "$QM" \
   && ! grep -B5 'self.moe_quant_config = self.get_fused_moe_quant_config(layer)' "$QM" \
        | grep -q 'def get_fused_moe_quant_config'; then
  # Need a check that the assignment is present at the END of process_weights_after_loading
  # (which is right before get_fused_moe_quant_config in the W8A8Int8MoEMethod class).
  if grep -q 'self.moe_quant_config = self.get_fused_moe_quant_config(layer)' "$QM"; then
    echo "[patches] #1 quark_moe.py already patched (skipping)"
  fi
fi
if ! grep -q '^\s*self\.moe_quant_config = self\.get_fused_moe_quant_config(layer)$' "$QM"; then
  python <<'PY'
import re, pathlib, sys
p = pathlib.Path(__import__("os").environ.get("VLLM_LIB",
       __import__("vllm").__file__.rsplit("/", 1)[0])
     ) / "model_executor/layers/quantization/quark/quark_moe.py"
src = p.read_text()
# Locate the QuarkW8A8Int8MoEMethod class' process_weights_after_loading method,
# specifically the spot RIGHT BEFORE its get_fused_moe_quant_config method.
pat = re.compile(
    r"(layer\.w13_weight_scale = torch\.nn\.Parameter\(\n"
    r"\s+max_w13_scales, requires_grad=False\n"
    r"\s+\)\n)"
    r"(\s+def get_fused_moe_quant_config\(\n"
    r"\s+self, layer: torch\.nn\.Module\n"
    r"\s+\) -> FusedMoEQuantConfig \| None:\n"
    r"\s+if self\.weight_qscheme == \"per_channel\" and not self\.static_input_scales:\n"
    r"\s+return int8_w8a8_moe_quant_config\()",
    re.M,
)
m = pat.search(src)
if not m:
    sys.stderr.write("[patches] #1 could not locate target — manual fix needed in quark_moe.py\n")
    sys.exit(1)
new = src[:m.end(1)] + "\n        self.moe_quant_config = self.get_fused_moe_quant_config(layer)\n\n" + src[m.start(2):]
p.write_text(new)
print("[patches] #1 quark_moe.py patched")
PY
else
  echo "[patches] #1 quark_moe.py already patched"
fi

# ──────────────────────────────────────────────────────────────────────────
# Patch 2: _get_config_dtype_str missing int8_w8a8 case (+ config_name pass-through)
# ──────────────────────────────────────────────────────────────────────────
CONF="$VLLM_LIB/model_executor/layers/fused_moe/config.py"
if grep -q '"int8_w8a8"' "$CONF" && grep -q 'use_int8_w8a8' "$CONF"; then
  echo "[patches] #2 config.py already patched"
else
  python <<'PY'
import re, pathlib
p = pathlib.Path(__import__("os").environ.get("VLLM_LIB",
       __import__("vllm").__file__.rsplit("/", 1)[0])
     ) / "model_executor/layers/fused_moe/config.py"
src = p.read_text()

# (a) add use_int8_w8a8 param to _get_config_dtype_str signature
src = re.sub(
    r"(def _get_config_dtype_str\(\n\s+dtype: torch\.dtype,\n\s+use_fp8_w8a8: bool = False,)",
    r"\1\n    use_int8_w8a8: bool = False,",
    src, count=1,
)
# (b) add elif branch
src = re.sub(
    r"(    if use_fp8_w8a8:\n        return \"fp8_w8a8\"\n)",
    r'\1    elif use_int8_w8a8:\n        return "int8_w8a8"\n',
    src, count=1,
)
# (c) pass it from config_name()
src = re.sub(
    r"(        return _get_config_dtype_str\(\n            use_fp8_w8a8=self\.use_fp8_w8a8,)",
    r"\1\n            use_int8_w8a8=self.use_int8_w8a8,",
    src, count=1,
)
p.write_text(src)
print("[patches] #2 config.py patched")
PY
fi

# ──────────────────────────────────────────────────────────────────────────
# Patch 3: humming_utils.py stale import
# ──────────────────────────────────────────────────────────────────────────
HU="$VLLM_LIB/model_executor/layers/quantization/utils/humming_utils.py"
if grep -q 'from vllm.model_executor.layers.fused_moe.routed_experts import' "$HU"; then
  sed -i 's|from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts|from vllm.model_executor.layers.fused_moe import RoutedExperts|' "$HU"
  echo "[patches] #3 humming_utils.py patched"
else
  echo "[patches] #3 humming_utils.py already patched"
fi

# ──────────────────────────────────────────────────────────────────────────
# Install the pre-tuned A100 MoE config (saves us a 30-min autotune on cold start)
# ──────────────────────────────────────────────────────────────────────────
CFG_DIR="$VLLM_LIB/model_executor/layers/fused_moe/configs"
SRC_DIR="$(dirname "$(readlink -f "$0")")/config"
for cfg in "$SRC_DIR"/E=256,N=512,device_name=*.json; do
  [ -f "$cfg" ] && cp "$cfg" "$CFG_DIR/" && echo "[patches] tuned MoE config installed: $(basename "$cfg")"
done

# ──────────────────────────────────────────────────────────────────────────
# Linker fix: nvrtc unversioned symlink (needed for any humming-kernels JIT)
# ──────────────────────────────────────────────────────────────────────────
NVIDIA_DIR="$VLLM_LIB/../nvidia"
if [ -d "$NVIDIA_DIR/cu13/lib" ] && [ ! -e "$NVIDIA_DIR/cu13/lib/libnvrtc.so" ]; then
  ln -sf libnvrtc.so.13 "$NVIDIA_DIR/cu13/lib/libnvrtc.so"
  ln -sf libnvrtc-builtins.so.13.0 "$NVIDIA_DIR/cu13/lib/libnvrtc-builtins.so"
  echo "[patches] nvrtc unversioned symlinks created"
fi

echo "[patches] DONE"