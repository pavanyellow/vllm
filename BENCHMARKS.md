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
