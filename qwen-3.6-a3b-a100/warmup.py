"""Send N synthetic warm-up requests until p50 TTFT stabilizes.

First few requests after server startup are slow:
  - Triton JIT compiles for unseen shapes
  - CUDA graphs may not have replayed at that exact size yet
  - HF tokenizer warm-up

This script blocks until either (a) N requests have completed AND the last
few are stable, or (b) timeout. Use it in container health probes.

Usage:
  python warmup.py --base-url http://localhost:8000 --api-key sk-... [--lengths 100,1000,4000]
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time

from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="qwen-3.6-a3b")
    p.add_argument("--lengths", default="100,1000,4000",
                   help="comma-sep approximate input token counts to warm up at")
    p.add_argument("--per-length", type=int, default=3,
                   help="warmup requests at each length")
    p.add_argument("--timeout", type=int, default=600,
                   help="max seconds to wait for server to accept requests")
    return p.parse_args()


def wait_for_server(client: OpenAI, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.models.list()
            print(f"[warmup] server is up", flush=True)
            return
        except Exception as e:
            print(f"[warmup] waiting for server: {type(e).__name__}", flush=True)
            time.sleep(5)
    raise TimeoutError(f"server not responsive within {timeout}s")


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    wait_for_server(client, args.timeout)

    lengths = [int(x) for x in args.lengths.split(",")]
    # Use plain ASCII filler so each char is ~one BPE token (close enough for warm-up)
    BASE = "the quick brown fox jumps over the lazy dog "
    for L in lengths:
        prompt = (BASE * (L // 9 + 1))[: L * 4]  # rough char→token ratio
        times = []
        for i in range(args.per_length):
            t0 = time.perf_counter()
            r = client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
            )
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            print(f"[warmup] L≈{L:>5} trial {i+1}: {dt:.1f} ms", flush=True)
        med = statistics.median(times)
        print(f"[warmup] L≈{L:>5} median: {med:.1f} ms (last={times[-1]:.1f})", flush=True)
    print("[warmup] DONE", flush=True)


if __name__ == "__main__":
    main()
