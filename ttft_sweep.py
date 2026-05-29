#!/usr/bin/env python3
"""TTFT vs. input-length sweep for a running vLLM server.

Measures time-to-first-token (TTFT) across a range of input lengths, to probe
where prefill moves from memory-bound (flat TTFT, dominated by the one-time
weight read) to compute-bound (TTFT rising linearly with tokens), and to expose
shape-quantization (sawtooth) effects.

Key properties:
  * Sends raw random token IDs via /v1/completions, so each prompt is unique
    => zero prefix-cache hits (we measure the warm compute path, not replays),
    and the exact input length is controlled with no tokenizer variance.
  * --concurrency C fires C requests together (thread pool) and times each;
    C=1 is the single-stream case.
  * TTFT = wall-clock from request send to the first streamed token chunk.

Usage:
    .venv/bin/python ttft_sweep.py                       # default 100..5000, C=1
    .venv/bin/python ttft_sweep.py --lengths 2048,4096,4224 --concurrency 2
    .venv/bin/python ttft_sweep.py --start 256 --stop 32000 --step 2048
"""

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

DEFAULT_LENGTHS = sorted(set(list(range(100, 5001, 300)) + [5000]))
VOCAB_LO, VOCAB_HI = 100, 150_000   # safe non-special token-id range


def build_prefix(length: int, cache_frac: float) -> list:
    # Deterministic shared prefix of round(cache_frac*length) tokens. Same tokens
    # every call => identical block hashes => prefix-cache HIT once primed. The
    # remaining suffix is fresh random per call => MISS (real prefill).
    plen = int(round(length * cache_frac))
    if plen <= 0:
        return []
    rng = random.Random(0xBEEF ^ length)   # stable across calls/runs for this length
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(plen)]


def prime(url: str, model: str, prefix: list) -> None:
    # Send the prefix once so its blocks land in the prefix cache before timing.
    if not prefix:
        return
    requests.post(url, json={"model": model, "prompt": prefix, "max_tokens": 1,
                             "temperature": 0.0, "ignore_eos": True}).raise_for_status()


def ttft_once(url: str, model: str, input_len: int, out_tokens: int,
              prefix: list) -> float:
    # prefix = cached portion (constant); suffix = fresh random => unique tail.
    suffix = random.choices(range(VOCAB_LO, VOCAB_HI), k=input_len - len(prefix))
    prompt_ids = prefix + suffix
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


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def measure(url, model, length, out_tokens, rounds, warm, concurrency, cache_frac):
    prefix = build_prefix(length, cache_frac)
    prime(url, model, prefix)   # cache the shared prefix before timing
    samples = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rnd in range(rounds):
            futs = [ex.submit(ttft_once, url, model, length, out_tokens, prefix)
                    for _ in range(concurrency)]
            res = [f.result() for f in futs]
            if rnd >= warm:
                samples.extend(res)
    return samples


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
    p.add_argument("--concurrency", type=int, default=1, help="requests in flight together")
    p.add_argument("--n", type=int, default=12, help="rounds per length")
    p.add_argument("--warm", type=int, default=2, help="discarded warmup rounds per length")
    p.add_argument("--out-tokens", type=int, default=5)
    p.add_argument("--cache-frac", type=float, default=0.0,
                   help="fraction of each prompt that is a primed, cached shared "
                        "prefix (0.0 = clean baseline, no cache hit)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.lengths:
        lengths = [int(x) for x in args.lengths.split(",")]
    elif args.start and args.stop and args.step:
        lengths = list(range(args.start, args.stop + 1, args.step))
    else:
        lengths = DEFAULT_LENGTHS

    print(f"concurrency={args.concurrency} cache_frac={args.cache_frac}")
    print(f"{'in_len':>7} {'TTFT_p50_ms':>12} {'TTFT_p90_ms':>12} {'TTFT_min_ms':>12}")
    for length in lengths:
        s = measure(args.url, args.model, length, args.out_tokens,
                    args.n, args.warm, args.concurrency, args.cache_frac)
        p50 = percentile(s, 0.50) * 1000.0
        p90 = percentile(s, 0.90) * 1000.0
        tmin = min(s) * 1000.0
        print(f"{length:>7} {p50:>12.1f} {p90:>12.1f} {tmin:>12.1f}", flush=True)


if __name__ == "__main__":
    main()
