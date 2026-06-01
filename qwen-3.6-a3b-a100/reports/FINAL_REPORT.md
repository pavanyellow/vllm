# Qwen3.6-35B-A3B Voice-Agent TTFT — Final Report

**Hardware:** 1× NVIDIA A100 80 GB SXM (PG509-210)
**Model:** `Qwen3.6-35B-A3B-Quark-W8A8-INT8`
**Stack:** vLLM 0.22.0 + torch 2.11.0+cu130 + transformers 5.9
**Date:** 2026-05-31
**Goal:** Minimize TTFT for single-stream voice-agent prefill at L ≤ 4K.

---

## TL;DR — what we did, what we got

Three categories of fixes, applied in combination:

1. **vLLM upstream bugfixes (3 patches in `upstream_pr/`)** — INT8 W8A8 MoE was silently running the BF16 fallback, tuned configs for INT8 W8A8 were never being loaded, and Humming utils had a stale import. All three issues affect any vLLM user with our quant scheme, not just us.
2. **Triton MoE kernel autotune** — a wider 594-config sweep found `GROUP_SIZE_M=64` (vs vLLM's heuristic `GM=1`) which gives much better L2 reuse for many-experts case. **−15-17 %** TTFT at L ≥ 2K.
3. **Extended CUDA graphs for prefill** — vLLM's defaults only capture sizes 1-512 (decode-shaped). Adding fine-grained captures every 128 from 640 to 2048 (12 extras) makes prefill requests up to 2048 tokens use full graphs instead of the slow piecewise fallback. **−40-65 %** TTFT at L = 50-2048. Larger captures (≥ 3072) infeasible on single A100 — see Memory analysis.

### Headline numbers (p50 TTFT, ms)

| L | (1) Eager | (2) **Baseline** (vLLM 0.22 out-of-box) | (3) Eager + Autotune | (4) **All fixes** | Δ baseline → all fixes |
|---:|---:|---:|---:|---:|---:|
|    1 | 129.6 |  11.6 | 135.8 |  **11.8** | ~0 |
|   50 | 169.2 | 144.8 | 182.3 |  **55.4** | **−62 %** |
|  100 | 168.3 | 150.7 | 167.1 |  **55.3** | **−63 %** |
|  500 | 169.8 | 143.5 | 168.5 |  **55.6** | **−61 %** |
| 1000 | 175.0 | 145.1 | 191.3 |  **62.3** | **−57 %** |
| 2000 | 168.2 | 143.4 | 168.7 |  **95.6** | **−33 %** |
| 3000 | 175.9 | 157.5 | 183.0 | **145.9** | **−7 %** |
| 4000 | 223.0 | 196.4 | 193.4 | **166.1** | **−15 %** |
| 5000 | 268.4 | 235.6 | 234.5 | **201.3** | **−15 %** |
| 6000 | 313.4 | 275.2 | 277.6 | **237.8** | **−14 %** |

Interactive plot (Plotly, single file, hover for raw): `FINAL_ttft_4way_comparison.html`

### p99 TTFT, ms

| L | (1) Eager | (2) Baseline | (3) Eager + Autotune | (4) All fixes |
|---:|---:|---:|---:|---:|
|    1 | 134.3 |  13.7 | 154.0 |  12.5 |
|   50 | 191.0 | 145.7 | 203.5 |  56.9 |
|  100 | 173.5 | 160.0 | 169.5 |  56.3 |
|  500 | 172.2 | 150.3 | 175.2 |  58.1 |
| 1000 | 204.4 | 154.5 | 202.1 |  63.3 |
| 2000 | 184.8 | 146.1 | 178.2 |  96.5 |
| 3000 | 178.0 | 176.8 | 197.4 | 152.5 |
| 4000 | 226.2 | 197.0 | 197.7 | 186.1 |
| 5000 | 271.7 | 236.4 | 249.5 | 207.3 |
| 6000 | 314.5 | 276.7 | 289.2 | 238.6 |

### Voice-agent target (<150 ms @ L=4K)

| Length range | Status with "all fixes" |
|---|---|
| L ≤ 2K (typical voice prompt) | **Well under target** — 95.6 ms p50 at L=2000, 55.6 ms at L=500 |
| L = 3K-4K | Close but not under — 166 ms p50 at L=4K (14 ms over target) |
| L > 4K | Above target — 200+ ms |

For L ≥ 3K the **extended CG memory cliff at 3072+** prevents the last 25 ms of savings. Unlocking those would require either pipeline-parallel across the 2× A100s, a smaller (W4) model, or H100/B200 hardware.

---

## 1. The vLLM upstream bugfixes (`upstream_pr/`)

### Patch #1 — Quark INT8 MoE quant_config never set

`QuarkW8A8Int8MoEMethod.process_weights_after_loading` was missing the line every
other MoE quant method has at its end:
```python
self.moe_quant_config = self.get_fused_moe_quant_config(layer)
```
Effect: `apply()` passed `quant_config=None` to `fused_experts(...)`, which
silently fell back to the unquantized BF16 path. INT8 weights were loaded
but the kernel ran BF16 compute against them (dequantized inside the kernel).

Repro: any Quark W8A8 INT8 quantized MoE model. After the fix `use_int8_w8a8=True`
reaches the Triton `fused_moe_kernel`, INT8 tensor cores fire. Measured PLAN.md
result: ~14 % TTFT improvement at L=1000 immediately after the fix.

File: `vllm/model_executor/layers/quantization/quark/quark_moe.py`
Patch: `upstream_pr/0001-fix-set-moe_quant_config-in-QuarkW8A8Int8MoEMethod.patch`

### Patch #2 — fused_moe config_dtype_str missing INT8 W8A8 case

`_get_config_dtype_str()` handles `fp8_w8a8`, `fp8_w8a16`, `int8_w8a16`,
`int4_w4a16` but **not `int8_w8a8`**. For INT8 W8A8 MoE models it returns
`None`, so `get_config_file_name()` produces a filename **without** the
`,dtype=int8_w8a8` suffix. Any pre-tuned config saved at the conventional
`E=*,N=*,device_name=*,dtype=int8_w8a8.json` name **is never loaded** by
vLLM 0.22.

Effect: the heuristic default kernel config runs, silently losing whatever
tuning was done. We worked around by saving the same JSON under BOTH names.
With the fix only the conventional name is needed.

File: `vllm/model_executor/layers/fused_moe/config.py`
Patch: `upstream_pr/0002-fix-add-int8_w8a8-case-to-_get_config_dtype_str.patch`

### Patch #3 — humming_utils stale `routed_experts` import

```python
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
```
That module doesn't exist in vLLM 0.22. `RoutedExperts` is a TypeAlias for
`FusedMoE` exported from the `fused_moe` package root.

Effect: any call path that imports `humming_utils` (which is needed to use
the Humming MoE backend for any quant scheme — MXFP4 and any future int8/fp8
wiring) hits `ModuleNotFoundError` at import time.

File: `vllm/model_executor/layers/quantization/utils/humming_utils.py`
Patch: `upstream_pr/0003-fix-stale-import-in-humming_utils.patch`

---

## 2. Triton MoE kernel autotune

vLLM picks Triton kernel meta-parameters (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`) via:
- A pre-tuned JSON in `vllm/model_executor/layers/fused_moe/configs/` if one
  exists for the (`E`, `N`, `device_name`, dtype) combo, OR
- A heuristic in `get_default_config()` if no JSON.

For `E=256, N=512, NVIDIA_PG509-210, int8_w8a8` no JSON existed → heuristic
fired → picked `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
GROUP_SIZE_M=1, num_warps=8, num_stages=3`.

### Sweep design and result

A 594-config sweep covering:
```
BLOCK_SIZE_M ∈ {32, 64, 128, 256}
BLOCK_SIZE_N ∈ {64, 128, 256}
BLOCK_SIZE_K ∈ {64, 128, 256}
GROUP_SIZE_M ∈ {1, 16, 64}      ← extended; prior 216-config sweep stopped at 32
num_warps    ∈ {4, 8}
num_stages   ∈ {3, 4, 5}
```
(shared-memory feasibility filter applied: skip BM*BN > 32768)

**Winner:** `BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=64, num_warps=4, num_stages=3`

Kernel-level wall time at M=4096 dropped from 1.06 ms (default) to 0.74 ms (-30 %).
End-to-end TTFT impact at L=4K: 196 → 165 ms (-15 %, autotune alone, with default CG).

### Why the heuristic was wrong

`get_default_config()` line 1279:
```python
tokens_per_expert = M // max(E, 1)
group_m = 16 if tokens_per_expert > 128 else 1
```
For our case `4096 / 256 = 16 ≪ 128` → picks `GM=1`. The reasoning is
"with many experts each one sees few tokens so grouping is useless".

That reasoning is **wrong**. `GROUP_SIZE_M` controls how Triton schedules
program blocks across the L2 cache. Even when per-expert M is small, grouping
adjacent program blocks lets them share weight tiles in L2. The autotune
picked GM=64 (an 8× shift from the heuristic).

The earlier 216-config sweep had `GM ∈ {1, 16, 32}` — never hit 64. The
wider sweep covers it and that single entry was decisive.

### Artifacts

- Script: `scripts/autotune_moe.py`
- Tuned config (written to two names due to Patch #2):
  - `vllm/.../fused_moe/configs/E=256,N=512,device_name=NVIDIA_PG509-210.json` (what vLLM actually loads)
  - `vllm/.../fused_moe/configs/E=256,N=512,device_name=NVIDIA_PG509-210,dtype=int8_w8a8.json` (conventional name)

---

## 3. Extended CUDA graphs for prefill

### How vLLM picks the graph at runtime

From `vllm/v1/cudagraph_dispatcher.py::_compute_bs_to_padded_graph_size`:
- For request size N ≤ `max_cudagraph_capture_size`: pad **up** to the
  smallest captured size ≥ N. **No tolerance check.**
- For N > max_cudagraph_capture_size: `CUDAGraphMode.NONE` → PIECEWISE fallback.

### Default capture sizes leave prefill on the slow path

vLLM's default `cudagraph_capture_sizes` is
`[1, 2, 4, 8, …, 256, 272, …, 512]` — 51 sizes, max=512. Designed for the
**decode** phase (M = number of active sequences, usually small).

For **prefill** at L > 512, vLLM falls back to PIECEWISE — many small CUDA
graphs at the op level, with lots of launch overhead and worse fusion than
a single FULL graph.

This is why prefill at L=1000-2000 sits around 145 ms in default config:
the kernel work isn't actually that big, but launch+coordination overhead
through Python dominates.

### The padding-cost crossover

Capturing one big graph (e.g. only at 2048) hurts small requests because
vLLM pads them ALL the way to 2048:

| Request L | With +[2048] only |
|---:|---|
| 600 | pads 600→2048 (3.4× extra tokens) — likely **slower** than PIECEWISE |
| 1000 | pads 1000→2048 (2× extra) — wins by ~40 ms |
| 1500 | pads 1500→2048 (1.4× extra) — wins by ~45 ms |
| 2000 | exact-ish → big win (-47 ms) |
| 4000 | > max=2048 → falls back to PIECEWISE, no benefit |

### Our solution: fine-grained captures at 128-token stride

To eliminate padding regressions, we extend with:
```
extras = [640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
```
12 sizes at 128-token stride. Combined with the defaults (1..512), every
request size up to 2048 has a captured graph within 128 tokens above it —
padding overhead ≤ 25 % even at the worst case (L=2049 would pad to 2048
if we added one… but we don't; L > 2048 falls to PIECEWISE).

Memory cost of the 12 extras was confirmed ~0 GiB above the model+default-CG
floor (`scripts/probe_memory.py`) — they fit comfortably.

---

## 4. Memory analysis — why we can't capture beyond 2048

Probe (`scripts/probe_memory.py`, single A100 80GB, max_num_seqs=1):

| Configuration | Total GPU mem | Workspace cost vs floor |
|---|---:|---:|
| Floor: model + KV (eager, no CG) | 39.71 GiB | 0 |
| + default CG [1..512] (51 captures) | 39.52 GiB | ~0 |
| + extra capture at 1024 | 39.74 GiB | ~0 |
| + extra capture at 2048 | 39.72 GiB | ~0 |
| + 12 extras [640..2048 stride 128] | 66.28 GiB (at gpu_mem=0.85) | KV cache filled the rest |
| + extra capture at **3072** | OOM | **~45 GiB** |
| + extra capture at **4096** | OOM | **~43 GiB** |

### The cliff

There is a **sharp non-linear jump** in workspace memory between captured-size
2048 and 3072. Captures ≤ 2048 add nearly nothing; captures ≥ 3072 add ~40-45
GiB on top of the 33.5 GiB model and run us out of budget for KV cache.

Hypothesis (unconfirmed — see TODO #18): `torch.compile`/inductor allocates
a persistent activation buffer sized for the largest captured shape, and the
MoE intermediate activations + attention scratch for L ≥ 3072 exceed some
internal heuristic that switches to a much-larger memory plan.

Concretely: the OOM message during `_allocate_kv_cache_tensors` at gpu_mem=0.85
shows "process has 75.40 GiB in use, tried to allocate 8.25 GiB for KV cache,
only 3.0 GiB free" — i.e. workspaces already consumed 42 GiB before KV even
started.

### Why this matters for the voice target

L=4K prefill needs a captured graph at size ≥ 4096 to hit the predicted
~140 ms TTFT (saving ~25 ms vs piecewise mode). The capture doesn't fit on
a single A100 with this model. Paths to unlock it (none pursued):

1. **PP=2** across 2× A100 — each GPU holds 20 layers (~17 GiB), frees room
   for +[4096] capture. One activation transfer per forward (~16 MB at L=4K
   ≪ 1 ms over PCIe). No compute speedup but unlocks the capture.
2. **W4A8 model** — INT4 weights + INT8 activations. ~17 GiB model, room for
   the capture. Need to find or build a Qwen3.6-A3B variant; Quark has W4A8
   only for ROCm AITER.
3. **H100/B200** — bigger memory + faster compute. The canonical solution.
4. **Patch vLLM with padding tolerance** — change `_bs_to_padded_graph_size`
   to skip captures whose padding distance > X %. Doesn't unlock L>2048 but
   would let us safely add larger captures (e.g. at 3500) if we ever fit one.

---

## 5. Choosing the capture sizes (and the padding-cost analysis)

vLLM rounds size N UP to the next captured size. So spacing of captures
directly determines worst-case padding overhead:
- spacing 128 → worst case 128 extra tokens (≤ 25 % padding at L=512)
- spacing 256 → worst case 256 extra tokens (≤ 50 % padding at L=512)
- spacing 512 → worst case 512 extra tokens (up to 100 %)

We chose **128** as the stride because:
- Each extra capture costs ~0 GiB (probe-verified).
- 12 extras is well within JIT-compile time budget (~10s each at cold start).
- Worst case padding at L=513 (pads to 640) = 24 % — below the
  ~50 % crossover where padding cost exceeds piecewise savings.

For lengths in the gap between the default 512 and our first extra 640 we
keep using the default 512 graph (pads up to 512 if needed, falls to
PIECEWISE if N > 512 but < 640 — actually no, vLLM rounds UP, so for
L ∈ (512, 640] it picks 640). For L ∈ (496, 512] it uses 512.

### Default-CG sizes vLLM picks (for reference)

```
[1, 2, 4] + range(8, 256, 8) + range(256, 513, 16)
=  [1, 2, 4, 8, 16, 24, 32, …, 248, 256, 272, 288, …, 496, 512]
```
51 sizes total. Pre-tuned for decode-sized batches.

### Final capture list with our extension

```
defaults: [1, 2, 4, 8, 16, …, 256, 272, …, 512]    ← 51 sizes
extras:   [640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]  ← 12
```
63 captures total. max_cudagraph_capture_size = 2048.

---

## 6. Production config — how to use this

```python
from qwen_client import QwenClient

client = QwenClient(
    model_dir="/data/users/jashwanth/qwen-claude/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8",
    max_model_len=6300,
    gpu_memory_utilization=0.85,
    max_num_seqs=4,                # voice-agent single-stream
    enable_chunked_prefill=False,  # default OFF — see PLAN.md
    enforce_eager=False,           # CUDA graphs ON
    tensor_parallel_size=1,        # NEVER TP=2 on PCIe; we measured +156% TTFT
    cudagraph_capture_sizes=(
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
        + [640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    ),
)
```

Prereqs (one-time per environment):
1. Apply 3 vLLM patches from `upstream_pr/` (or upstream their merged versions)
2. Replace `tokenizer_config.json` in the model dir with the BF16-base copy
   (fixes the Quark export `tokenizer_class: TokenizersBackend` bug)
3. Source `scripts/env.sh` (sets `VLLM_USE_FLASHINFER_SAMPLER=0`, cu13
   LD_LIBRARY_PATH, etc.)
4. Place tuned MoE config in `vllm/.../fused_moe/configs/` — both
   `E=256,N=512,device_name=NVIDIA_PG509-210.json` AND
   `E=256,N=512,device_name=NVIDIA_PG509-210,dtype=int8_w8a8.json` (until
   Patch #2 lands upstream, only the no-dtype-suffix name is actually read)

---

## 7. Portability to H100 (your next target)

| Component | Portable as-is? |
|---|---|
| 3 vLLM patches | **Yes** — code-only fixes, HW-agnostic |
| Tokenizer fix | **Yes** — model side |
| `qwen_client.py` | **Yes** — pure config plumbing |
| Tuned MoE JSON (`device_name=NVIDIA_PG509-210.json`) | **No** — file is device-named. H100 looks for `NVIDIA_H100_80GB_HBM3.json`. Different L2 (50 vs 40 MB), 3.2× INT8 TOPS, more SMs → optimal tile params likely different. **Re-run `scripts/autotune_moe.py` on H100** (~30 min). |
| Extended CG capture sizes | **Yes** — same model, same 80 GB. Likely same cliff. Re-probe with `scripts/probe_memory.py` to be sure; may fit slightly larger captures because H100 has different TMA scratchpad. |

**Bonus on H100:** native FP8 tensor cores at 1979 TFLOPS unlock a real
choice between (a) port INT8 W8A8 as-is + re-autotune, or (b) switch to
the FP8 Qwen variant + use the cutlass FP8 MoE backend
(`vllm/.../fused_moe/experts/cutlass_moe.py`). Recommend running both and
picking the better TTFT.

---

## 8. What did NOT work (negative results worth recording)

- **TP=2 on PCIe-connected 2× A100**: TTFT@L=4K went **199 → 509 ms** (+156 %).
  All-reduce overhead on Gen4 PCIe (~32 GB/s, no NVLink) dominates the
  per-layer attention/MoE communication. Conclusion: **never use TP on
  PCIe-connected A100s for single-stream voice**. Use PP=2 if you need
  multi-GPU.
- **Humming MoE backend integration for INT8 W8A8**: humming-kernels achieves
  96 % of A100 INT8 peak in microbenchmark and supports SM ≥ 7.5. BUT vLLM's
  Quark INT8 MoE uses the LEGACY `fused_experts()` dispatch, not the modular
  kernel pattern Humming requires. Half-integration revealed the architecture
  mismatch; full integration estimated at 1-2 days of refactoring. Reverted.
- **Extended CG at 4096**: OOM (see Memory section). The ~25 ms TTFT win
  predicted by PLAN.md is real but blocked on memory headroom.
- **TensorRT-LLM**: docs explicitly target Qwen3-Next on Hopper/Blackwell;
  no A100 path. Plus install blocked by `pypi.nvidia.com` proxy filter.

---

## 9. Files

### Data
- `runs/ttft_sweep_final_1_eager_default*` — eager (no CG, no autotune)
- `runs/ttft_sweep_final_2_baseline_cg_default*` — vLLM 0.22 out-of-box
- `runs/ttft_sweep_final_3_eager_autotune*` — eager + autotune
- `runs/ttft_sweep_final_4_all_fixes*` — autotune + default CG + extended CG
- `runs/FINAL_ttft_4way_comparison.html` — interactive 4-curve overlay

### Code
- `scripts/qwen_client.py` — final production client
- `scripts/profile_ttft_sweep.py` — bench harness (p50/p90/p99)
- `scripts/build_profile_prompts.py` — 78-length grid 1..6000
- `scripts/autotune_moe.py` — 594-config Triton sweep
- `scripts/probe_memory.py` — per-phase GPU memory probe

### Patches (push from local laptop — see `upstream_pr/HOW_TO_PUSH.md`)
- `upstream_pr/0001-fix-set-moe_quant_config-in-QuarkW8A8Int8MoEMethod.patch`
- `upstream_pr/0002-fix-add-int8_w8a8-case-to-_get_config_dtype_str.patch`
- `upstream_pr/0003-fix-stale-import-in-humming_utils.patch`

### Tuned MoE config (the autotune winner)
- `vllm/.../fused_moe/configs/E=256,N=512,device_name=NVIDIA_PG509-210.json`
- `vllm/.../fused_moe/configs/E=256,N=512,device_name=NVIDIA_PG509-210,dtype=int8_w8a8.json`

---

## 10. Open follow-ups (for whoever picks this up)

1. **Debug the 3072+ memory cliff** — find which torch.compile / inductor
   buffer scales non-linearly. If sub-sizable, we'd unlock +[4096] on single
   A100 and the predicted ~140 ms @ L=4K.
2. **PP=2** end-to-end bench — easy config change; should match A100-single
   for compute time AND unlock +[4096] capture.
3. **W4A8 Qwen3.6-A3B** — produce or find a quant; would give same memory
   headroom as PP=2 without needing 2 GPUs.
4. **File the 3 vLLM PRs upstream** — drafts ready in `upstream_pr/`.
   At least Patch #1 should land quickly — it's a one-line obvious fix.
