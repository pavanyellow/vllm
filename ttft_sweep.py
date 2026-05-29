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
from requests.adapters import HTTPAdapter

DEFAULT_LENGTHS = sorted(set(list(range(100, 5001, 300)) + [5000]))
VOCAB_LO, VOCAB_HI = 100, 150_000   # safe non-special token-id range


def make_session(pool_size: int) -> requests.Session:
    # Persistent session => HTTP keep-alive, so we reuse one warm TCP connection
    # per worker instead of paying a fresh connect/handshake on every request.
    # pool_maxsize >= concurrency so concurrent threads don't contend for sockets.
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def ttft_once(session: requests.Session, url: str, model: str,
              input_len: int, out_tokens: int) -> float:
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
    with session.post(url, json=payload, stream=True) as r:
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


def measure(session, url, model, length, out_tokens, rounds, warm, concurrency):
    samples = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for rnd in range(rounds):
            futs = [ex.submit(ttft_once, session, url, model, length, out_tokens)
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
    return p.parse_args()


def main():
    args = parse_args()
    if args.lengths:
        lengths = [int(x) for x in args.lengths.split(",")]
    elif args.start and args.stop and args.step:
        lengths = list(range(args.start, args.stop + 1, args.step))
    else:
        lengths = DEFAULT_LENGTHS

    session = make_session(args.concurrency)

    print(f"concurrency={args.concurrency}")
    print(f"{'in_len':>7} {'TTFT_p50_ms':>12} {'TTFT_p90_ms':>12} {'TTFT_min_ms':>12}")
    for length in lengths:
        s = measure(session, args.url, args.model, length, args.out_tokens,
                    args.n, args.warm, args.concurrency)
        p50 = percentile(s, 0.50) * 1000.0
        p90 = percentile(s, 0.90) * 1000.0
        tmin = min(s) * 1000.0
        print(f"{length:>7} {p50:>12.1f} {p90:>12.1f} {tmin:>12.1f}", flush=True)


if __name__ == "__main__":
    main()
