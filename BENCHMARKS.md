# Qwen/Qwen3.6-35B-A3B-FP8 — single-stream TTFT benchmarks

## Setup

- Model: `Qwen/Qwen3.6-35B-A3B-FP8` (served as `Qwen/Qwen3.6-35B-A3B`), FP8.
- Hardware: 1× NVIDIA H100 80GB.
- vLLM `0.1.dev17053+g7e53283b1`, `torch 2.11.0+cu128`, CUDA 12.8, Python 3.12.
- Server flags: `--max-model-len 32768 --gpu-memory-utilization 0.90
  --enable-prefix-caching --enable-chunked-prefill --language-model-only
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`.
- Scheduler: `max_num_batched_tokens=8192`. Inputs longer than 8192 tokens are
  prefilled in `ceil(input_len / 8192)` chunks.

## Methodology

- Concurrency 1 (one request in flight).
- Prompts are unique random token IDs (range 100–150000) sent as raw token IDs
  via `/v1/completions` → exact input length, and 0% prefix-cache hit rate
  (verified in server logs).
- Sampling: `max_tokens` small, `temperature=0`, `ignore_eos=true`, `stream=true`.
- TTFT = wall-clock from request send to first streamed token chunk.
- Section A produced by [`ttft_sweep.py`](./ttft_sweep.py): 10 requests per length,
  first 2 discarded as per-shape warmup, 5 output tokens.
- Section B produced by `vllm bench serve` (`--dataset-name random
  --random-output-len 1 --random-range-ratio 0 --random-prefix-len 0
  --max-concurrency 1 --num-prompts 20`).

## A. Input-length sweep, 256 → 32,000 tokens

| input tokens | TTFT p50 (ms) | TTFT min (ms) | prefill tok/s (len/TTFT) |
|---:|---:|---:|---:|
| 256    | 28.3  | 26.9  | 9,051  |
| 512    | 33.4  | 30.1  | 15,331 |
| 1,024  | 85.7  | 83.0  | 11,955 |
| 1,536  | 106.8 | 106.6 | 14,377 |
| 2,048  | 159.4 | 159.0 | 12,852 |
| 3,072  | 162.9 | 161.9 | 18,855 |
| 4,096  | 166.0 | 164.9 | 24,680 |
| 5,120  | 171.1 | 167.8 | 29,927 |
| 6,144  | 174.0 | 170.9 | 35,305 |
| 8,192  | 178.8 | 177.4 | 45,815 |
| 10,240 | 264.5 | 257.0 | 38,711 |
| 12,288 | 280.7 | 268.9 | 43,779 |
| 14,336 | 304.1 | 302.8 | 47,149 |
| 16,384 | 399.6 | 396.1 | 41,000 |
| 20,480 | 436.6 | 430.9 | 46,911 |
| 24,576 | 535.0 | 527.8 | 45,940 |
| 28,672 | 610.1 | 607.9 | 46,996 |
| 32,000 | 695.8 | 694.0 | 45,991 |

## B. Cross-check — `vllm bench serve`, 2k → 8k (concurrency 1, 1 output token)

| input tokens | TTFT mean (ms) | TTFT p50 (ms) | P90 (ms) | P99 (ms) |
|---:|---:|---:|---:|---:|
| 2,048 | 186.4 | 186.3 | 193.6 | 207.1 |
| 3,072 | 208.5 | 195.5 | 206.2 | 417.8 |
| 4,096 | 215.2 | 215.7 | 230.0 | 233.8 |
| 8,192 | 225.4 | 217.7 | 250.4 | 293.5 |

## C. Dense single-stream sweeps (concurrency 1)

Methodology as Section A (concurrency 1, unique random token-id prompts, 0%
prefix-cache hit verified, TTFT = send → first streamed token), with finer
length steps: `ttft_sweep.py --lengths ... --n 20 --warm 4` (20 rounds per
length, first 4 discarded; 5 output tokens).

### C.1 — 640 → 1536

| input tokens | TTFT p50 (ms) | TTFT min (ms) |
|---:|---:|---:|
| 640   | 82.4  | 81.6  |
| 768   | 82.3  | 81.5  |
| 896   | 82.0  | 81.5  |
| 960   | 83.0  | 82.0  |
| 1,024 | 83.0  | 82.1  |
| 1,152 | 100.7 | 98.6  |
| 1,280 | 100.2 | 99.3  |
| 1,408 | 104.7 | 103.5 |
| 1,536 | 107.0 | 106.5 |

### C.2 — 1664 → 2560

| input tokens | TTFT p50 (ms) | TTFT min (ms) |
|---:|---:|---:|
| 1,664 | 160.9 | 158.5 |
| 1,792 | 158.2 | 158.0 |
| 1,920 | 159.2 | 158.2 |
| 2,048 | 160.5 | 157.6 |
| 2,176 | 101.8 | 101.3 |
| 2,304 | 102.7 | 102.0 |
| 2,432 | 108.5 | 105.6 |
| 2,560 | 108.9 | 107.6 |

### C.3 — 3584 → 4608

| input tokens | TTFT p50 (ms) | TTFT min (ms) |
|---:|---:|---:|
| 3,584 | 111.7 | 110.7 |
| 3,712 | 164.3 | 163.2 |
| 3,840 | 165.7 | 163.3 |
| 3,968 | 166.2 | 164.2 |
| 4,096 | 165.4 | 164.4 |
| 4,224 | 94.4  | 92.8  |
| 4,352 | 108.3 | 107.7 |
| 4,480 | 110.1 | 109.7 |
| 4,608 | 113.5 | 113.0 |

## D. Dense sweeps at concurrency 2

Methodology as Section C, with `ttft_sweep.py --concurrency 2`: each round fires
2 equal-length requests together via a thread pool, TTFT recorded per request
(up to 20 rounds × 2 = 40 samples per length, first 4 rounds discarded). 0%
prefix-cache hit verified.

### D.1 — 1664 → 2560

| input tokens | TTFT p50 (ms) | TTFT p90 (ms) | TTFT min (ms) |
|---:|---:|---:|---:|
| 1,664 | 241.1 | 247.9 | 235.8 |
| 1,792 | 242.2 | 252.6 | 236.3 |
| 1,920 | 243.2 | 251.0 | 236.8 |
| 2,048 | 242.6 | 253.8 | 236.3 |
| 2,176 | 184.2 | 192.0 | 178.1 |
| 2,304 | 185.5 | 192.9 | 178.5 |
| 2,432 | 185.3 | 192.1 | 177.7 |
| 2,560 | 186.3 | 192.0 | 178.2 |

### D.2 — 3584 → 4608

| input tokens | TTFT p50 (ms) | TTFT p90 (ms) | TTFT min (ms) |
|---:|---:|---:|---:|
| 3,584 | 192.3 | 202.8 | 181.5 |
| 3,712 | 252.1 | 262.6 | 240.3 |
| 3,840 | 252.7 | 262.5 | 241.5 |
| 3,968 | 252.0 | 259.5 | 242.2 |
| 4,096 | 252.6 | 261.6 | 242.1 |
| 4,224 | 179.7 | 190.9 | 171.4 |
| 4,352 | 194.4 | 204.4 | 184.1 |
| 4,480 | 194.7 | 201.2 | 184.6 |
| 4,608 | 197.8 | 206.3 | 184.4 |
