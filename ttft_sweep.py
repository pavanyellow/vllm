#!/usr/bin/env python3
"""Single-stream TTFT vs. input-length sweep for a running vLLM server.

Measures time-to-first-token (TTFT) at concurrency 1 across a range of input
lengths, to probe where prefill moves from memory-bound (flat TTFT, dominated
by the one-time weight read) to compute-bound (TTFT rising linearly with tokens).

Key properties:
  * Sends raw random token IDs via /v1/completions, so each prompt is unique
    => zero prefix-cache hits (we measure the warm compute path, not replays),
    and the exact input length is controlled with no tokenizer variance.
  * One request in flight at a time (concurrency 1).
  * TTFT = wall-clock from request send to the first streamed token chunk.

Reports per length: median TTFT, min TTFT, and effective prefill throughput
(input_len / TTFT). In the memory-bound region throughput rises with length;
once compute-bound it plateaus.

Usage:
    # with the server already running on :8000
    .venv/bin/python ttft_sweep.py                       # default 100..5000 sweep
    .venv/bin/python ttft_sweep.py --lengths 256,2048,8192,16384,32000
    .venv/bin/python ttft_sweep.py --start 256 --stop 32000 --step 2048
"""

import argparse
import random
import time

import requests

DEFAULT_LENGTHS = sorted(set(list(range(100, 5001, 300)) + [5000]))
VOCAB_LO, VOCAB_HI = 100, 150_000   # safe non-special token-id range


def ttft_once(url: str, model: str, input_len: int, out_tokens: int) -> float:
    # fresh random token ids each call => unique prompt => no prefix-cache hit
    prompt_ids = random.choices(range(VOCAB_LO, VOCAB_HI), k=input_len)
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": out_tokens,
        "stream": True,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode() if isinstance(line, bytes) else line
            if s.startswith("data:"):
                data = s[5:].strip()
                if data and data != "[DONE]":
                    return time.perf_counter() - t0   # first token chunk = TTFT
    return time.perf_counter() - t0


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--lengths", default=None,
                   help="comma-separated input lengths (overrides --start/--stop/--step)")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--n", type=int, default=12, help="requests per length")
    p.add_argument("--warm", type=int, default=2, help="discarded warmup requests per length")
    p.add_argument("--out-tokens", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.lengths:
        lengths = [int(x) for x in args.lengths.split(",")]
    elif args.start and args.stop and args.step:
        lengths = list(range(args.start, args.stop + 1, args.step))
    else:
        lengths = DEFAULT_LENGTHS

    print(f"{'in_len':>7} {'TTFT_p50_ms':>12} {'TTFT_min_ms':>12} {'prefill_tok/s':>14}")
    for length in lengths:
        samples = []
        for i in range(args.n):
            t = ttft_once(args.url, args.model, length, args.out_tokens)
            if i >= args.warm:
                samples.append(t)
        p50 = median(samples) * 1000.0
        tmin = min(samples) * 1000.0
        tput = length / (p50 / 1000.0)
        print(f"{length:>7} {p50:>12.1f} {tmin:>12.1f} {tput:>14.0f}", flush=True)


if __name__ == "__main__":
    main()
