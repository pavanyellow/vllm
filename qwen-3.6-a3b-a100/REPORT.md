# Qwen3.6-35B-A3B W8A8 INT8 — TTFT Benchmark Report

**Hardware:** 1× NVIDIA A100 80GB SXM (`NVIDIA_A100-SXM4-80GB`)
**Model:** `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8` (33.5 GB, INT8 W8A8)
**Stack:** vLLM 0.22.0 + torch 2.11.0+cu130 + 3 upstream patches + MoE autotune + extended CUDA graphs
**Date:** 2026-06-01
**Benchmark:** In-process vLLM, single-stream, `max_tokens=1`, greedy, 78 lengths × 5-10 prompts each

---

## Headline TTFT (p50, in-process)

| Input Length | TTFT p50 | TTFT p90 | Notes |
|---:|---:|---:|---|
| 1 | **9 ms** | 17 ms | Only 8/256 experts touched |
| 50 | **33 ms** | 35 ms | CUDA graph region |
| 100 | **33 ms** | 39 ms | Overhead plateau |
| 500 | **41 ms** | 41 ms | |
| 1000 | **57 ms** | 57 ms | |
| 2000 | **88 ms** | 88 ms | Sub-100ms |
| 2400 | **99 ms** | 100 ms | Last sub-100ms point |
| 3000 | **119 ms** | 120 ms | Linear compute region |
| 4000 | **152 ms** | 152 ms | |
| 5000 | **187 ms** | 187 ms | |
| 6000 | **223 ms** | 224 ms | |

**Verified with two independent runs (seed=42, seed=99) — results within ±2ms.**

**Production (HTTP API) adds ~30-40ms** overhead from FastAPI + SSE streaming + JSON parsing.

---

## Optimizations Applied

### 1. vLLM upstream bugfixes (3 patches)

- **Patch #1 — Quark INT8 MoE quant_config never set:** `process_weights_after_loading()` was missing `self.moe_quant_config = self.get_fused_moe_quant_config(layer)`. INT8 weights were loaded but the kernel silently fell back to BF16 compute. Fix: ~14% TTFT improvement.
- **Patch #2 — `_get_config_dtype_str` missing `int8_w8a8`:** Pre-tuned MoE kernel configs were never loaded for INT8 W8A8 models. Fix: enables autotune config loading.
- **Patch #3 — `humming_utils` stale import:** `from vllm...routed_experts import RoutedExperts` pointed to a non-existent module in vLLM 0.22. Fix: corrects the import path.

### 2. MoE kernel autotune (`GROUP_SIZE_M=64`)

594-config sweep over Triton `fused_moe_kernel` meta-parameters. Key finding: `GROUP_SIZE_M=64` gives significantly better L2 cache reuse for the 256-expert MoE than vLLM's default heuristic (`GM=1`).

**Important:** The per-kernel autotune (which tests single kernel calls) found `GM=16` optimal on `A100-SXM4-80GB`. However, end-to-end benchmarks showed `GM=64` is **~7ms faster at L=4000** because L2 cache benefits compound across the 80 kernel calls per forward pass (40 layers × 2 FFNs). We override to `GM=64`.

### 3. Extended CUDA graphs for prefill

vLLM defaults capture CUDA graphs for sizes 1-512 (decode-shaped). We add 12 prefill-sized captures at stride 128 from 640 to 2048. This eliminates the slow piecewise fallback for prefill requests up to 2048 tokens.

Impact: **−40-65% TTFT at L=50-2048.** Captures beyond 2048 cause OOM on single A100 80GB.

---

## Improvement vs Previous Run

| Length | Previous (PG509-210, vLLM 0.19) | Current (SXM4-80GB, vLLM 0.22) | Improvement |
|---:|---:|---:|---:|
| 500 | 55.6 ms | **40.7 ms** | **−27%** |
| 1000 | 62.3 ms | **56.7 ms** | **−9%** |
| 2000 | 95.6 ms | **87.6 ms** | **−8%** |
| 4000 | 166.1 ms | **151.3 ms** | **−9%** |
| 6000 | 237.8 ms | **223.3 ms** | **−6%** |

Improvements come from vLLM 0.22 kernel scheduling + GM=64 override.

---

## Roofline Analysis

```
Memory floor  = 33.5 GB / (0.85 × 2.0 TB/s) = 19.7 ms
Compute slope = 2 × 3B / (0.80 × 312 TFLOPS) = 24.0 µs/token
Ridge point   ≈ 819 tokens
```

Three regimes:
1. **L=1:** Only 8 experts touched → less weight read → 9ms
2. **L=2-400:** Overhead plateau at ~33ms (CUDA graph launch overhead)
3. **L≥500:** Linear compute ramp at ~3.5ms per 100 tokens

---

## Variance

Extremely tight — p90 is within 1-2ms of p50 at every length above L=10. Standard deviation < 1ms for most lengths. This indicates stable, deterministic kernel execution with no scheduling jitter.

---

## Production Serving Notes

- **HTTP API overhead:** ~30-40ms (FastAPI + SSE + JSON). Production TTFT = in-process + 30-40ms.
- **TP=2 on PCIe:** Do NOT use. TTFT at L=4K goes 199 → 509ms (+156%). Only use TP=2 on NVLink.
- **Memory cliff at 3072+:** CUDA graph captures beyond 2048 tokens OOM on single A100 80GB. Would need PP=2, W4 quantization, or H100/B200 hardware.
- **GM=64 vs autotune default:** Always override to GM=64 for this model. The single-kernel autotune picks GM=16 which is ~7ms slower end-to-end.

---

## Files

| File | Description |
|---|---|
| `setup.sh` | One-shot installer (vLLM 0.22 + patches + model download + autotune) |
| `start_server.sh` | vLLM server launcher with extended CUDA graphs |
| `sweep_inprocess.py` | In-process TTFT sweep (no HTTP overhead) |
| `benchmark.py` | HTTP API benchmark (production-realistic) |
| `make_sweep_report.py` | Generates Plotly HTML report from sweep data |
| `autotune_moe.py` | 594-config Triton kernel sweep |
| `apply_vllm_patches.sh` | Applies 3 upstream bugfixes |
| `reports/ttft_sweep_inprocess.html` | Interactive TTFT scaling curve (Plotly) |
| `REPORT.md` | This file |

---

## Reproducing

```bash
# On a RunPod A100 80GB pod (or any A100 80GB):
git clone https://github.com/pavanyellow/vllm.git -b serve-qwen36-a100
cd vllm/qwen-3.6-a3b-a100
bash setup.sh                    # ~20 min first time (downloads model, patches, autotunes)

# In-process benchmark:
source /app/venv/bin/activate && source env.sh
python sweep_inprocess.py --out /workspace/sweep_results
python make_sweep_report.py --input /workspace/sweep_results/per_call.csv --out /workspace/sweep_results

# Production server:
bash start_server.sh             # ~3 min to ready
python benchmark.py --input sweep_prompts.jsonl --out /workspace/bench --stream --max-tokens 1 --warmup 5
```
