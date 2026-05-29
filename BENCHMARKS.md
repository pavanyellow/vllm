# Qwen/Qwen3.6-35B-A3B-FP8 — single-stream TTFT benchmarks

Time-to-first-token (TTFT) at **concurrency 1** on a single **NVIDIA H100 80GB**,
vLLM `0.1.dev17053+g7e53283b1`, FP8 weights. Server launched as documented in
[`README.md`](./README.md) (`--max-model-len 32768`, prefix caching + chunked
prefill on, thinking disabled).

**Methodology**

- Every prompt is unique random tokens → **0% prefix-cache hit rate** (verified in
  server logs), so each request does a full prefill — the warm compute path, not
  cache replays.
- Single request in flight (concurrency 1). Warmup requests discarded.
- TTFT = wall-clock from request send to first streamed token.

---

## A. Input-length sweep, 100 → 5000 tokens

Produced by [`ttft_sweep.py`](./ttft_sweep.py) (raw token IDs via `/v1/completions`,
5 output tokens, 12 requests/length, first 2 discarded).

| input tokens | TTFT p50 (ms) | TTFT min (ms) | prefill tok/s (len/TTFT) |
|---:|---:|---:|---:|
| 100  | 24.4  | 24.1  | 4,096  |
| 400  | 28.8  | 28.2  | 13,911 |
| 700  | 81.8  | 81.5  | 8,555  |
| 1000 | 83.7  | 82.4  | 11,945 |
| 1300 | 102.6 | 101.5 | 12,676 |
| 1600 | 161.7 | 158.4 | 9,897  |
| 1900 | 163.7 | 157.9 | 11,606 |
| 2200 | 104.9 | 101.7 | 20,973 |
| 2500 | 110.2 | 108.2 | 22,687 |
| 2800 | 162.5 | 161.0 | 17,229 |
| 3100 | 164.2 | 161.2 | 18,874 |
| 3400 | 107.4 | 106.4 | 31,670 |
| 3700 | 165.3 | 162.4 | 22,378 |
| 4000 | 165.9 | 163.6 | 24,105 |
| 4300 | 109.1 | 107.8 | 39,407 |
| 4600 | 114.4 | 113.7 | 40,220 |
| 4900 | 166.7 | 166.1 | 29,396 |
| 5000 | 167.8 | 167.3 | 29,791 |

**Read.** No clean memory→compute knee appears in this range. TTFT stays low
(≤ ~170 ms) and **oscillates between a ~110 ms band and a ~165 ms band**. Within
each length the spread is tight (min ≈ p50), so this is a *deterministic per-shape*
effect — most likely CUDA-graph / attention-kernel tile alignment, not a roofline
transition. Prefill remains weight-load-dominated (memory-bound) through 5k for
this 3B-active hybrid (linear-attention + 1-in-4 full-attention) MoE; the
compute-bound regime is beyond 5k tokens. Sweeping higher (16k–32k) would be
needed to locate the actual transition.

## B. Cross-check via `vllm bench serve` (random dataset)

Same server, `--max-concurrency 1`, `--random-output-len 1`, `--random-range-ratio 0`,
20 prompts/length.

| input tokens | TTFT mean (ms) | TTFT p50 (ms) | P90 | P99 |
|---:|---:|---:|---:|---:|
| 2048 | 186.4 | 186.3 | 193.6 | 207.1 |
| 3072 | 208.5 | 195.5 | 206.2 | 417.8 |
| 4096 | 215.2 | 215.7 | 230.0 | 233.8 |
| 8192 | 225.4 | 217.7 | 250.4 | 293.5 |

These run ~60–80 ms higher than Table A at comparable lengths because the
`vllm bench serve` OpenAI client path carries more per-request overhead (request
construction, response handling) than the lean raw-token client in `ttft_sweep.py`.
Both agree on the qualitative result: **TTFT is nearly flat vs. input length at
C=1** (only ~+30 ms from 2k→8k in Table B), confirming memory-bound prefill.
