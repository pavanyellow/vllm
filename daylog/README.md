# Serving Qwen3.6-35B-A3B-FP8 on H100 80GB — status

Empirical perf work on serving `Qwen/Qwen3.6-35B-A3B-FP8` (hybrid gated-deltanet +
attention MoE; 40 layers = 30 linear-attn + 10 full-attn, 256 experts top-8, ~3B
active) with vLLM 0.22.0. Target workload: single prompts ~3k–5k tokens, low
concurrency, TTFT-sensitive. Full method/empirics in the dated daylogs.

## Default launch (recommended for the 3k–5k, low-concurrency TTFT path)

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --max-num-batched-tokens 8192 \
  --block-size 8192 \
  --max-num-seqs 8
```
Run from a directory **outside** a vLLM source checkout (a local `vllm/` package
shadows the installed `vllm._C`). Client: `python ttft_sweep.py` (needs only
`requests`).

### Why the two non-stock flags
- **`--block-size 8192`** is the headline win (E9). Stock derives a **1056-token**
  attention block (smallest that makes the attention page ≥ the fixed mamba state
  page). Chunked prefill then splits any non-1056-multiple prompt into
  `floor(L/1056)·1056 + remainder` = an extra full 40-layer pass → up to **~2×
  TTFT** on typical prompts. block=8192 makes every prompt ≤8192 a single chunk →
  flat ~88–103 ms across 3k–5k (vs 95–180 ms sawtooth).
- **`--max-num-seqs 8`** is the price of block=8192. The unified allocator forces
  `mamba_page == attention_page`, so an 8192 block inflates each mamba slot 1056→8192
  (~6.8×, ~41 MiB→320 MiB). Capping concurrent sequences bounds that cache (≈8×320 MiB
  = 2.5 GiB) so it fits 80 GB; uncapped it OOMs at init. `--max-model-len 8192` also
  trims attention KV (we don't need 32k for this path).

### Concurrency caveat / when to revert
block=8192 trades KV capacity for TTFT: KV tokens 1.39M (@1056) → 269k (@8192),
max concurrency ~33×@8k. Great at low concurrency; **if serving many concurrent
sequences, the stock auto block (1056, full concurrency) may win** — re-baseline
under real load (open TODO). To revert: drop `--block-size`/`--max-num-seqs`.

## Where we are
- ✅ **E9 — block=8192 removes the chunked-prefill sawtooth (~2× TTFT).** Kept.
- ◻️ **E10 — per-component profile vs H100 roofline.** Σ GPU-busy @L=4200 = 66 ms vs
  ~15 ms roofline (4.4×). Gap = (1) gated-deltanet scan ~140× over E6's O(L) model,
  (2) ~20 ms (30%) overhead with no roofline term (FP8 quant + routing + norm),
  (3) real matmuls 2.4–2.9× over roof. MoE expert GEMM is the largest bucket but is
  **memory-bound** on the 32 GB weight read (9.6 ms floor on H100 HBM3).
- ❌ **E11 — tuning the Triton MoE config: null** (memory-bound; config can't help).
  Tuned JSON committed anyway (silences the "default MoE config" warning).
- ◻️ **E12 — forcing `--moe-backend deep_gemm`: ~8% faster matmul but ~1.4% net**
  (grouping/transpose overhead). Reverted to Triton (auto).

### Open (next)
- E6 roofline for chunked gated-deltanet (real FLOPs, not O(L)) — the ~140× miss.
- Attack the ~20 ms fusable overhead (quant/route/norm) — 30% of GPU-busy.
- Concurrency cost of block=8192 under real load (max-num-seqs vs OOM / KV drop).
- Two-tier remainder threshold (332 vs 528) from E8.

## Files
- `daylog/2026-05-29.md` — B200 183GB (E1–E7): TTFT sweep, prefix cache, prefill
  scaling, roofline, kernel attribution. Separate hardware reference.
- `daylog/2026-05-30-h100.md` — H100 80GB (E8–E12): this run.
- `ttft_sweep.py` — TTFT vs input-length sweep (raw token-ids, `--cache-frac`).
- `prof_prefill.py` — drive torch profiler around one prefill, bucket device-kernel
  time by component.
- `vllm/model_executor/layers/fused_moe/configs/E=256,N=512,…H100…fp8_w8a8,block_shape=[128,128].json`
  — tuned MoE config (E11; perf-neutral, removes warning).
```
Box: H100 80GB HBM3, vLLM 0.22.0, torch 2.11.0+cu130, CUDA 13.0, py 3.12.
```
