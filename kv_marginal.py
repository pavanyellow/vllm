#!/usr/bin/env python3
"""Marginal-token TTFT on the KV/prefix-cache path.

Holds a fixed PREFIX (default 4000 tokens) constant and primed into the prefix
cache, then measures TTFT as a function of how many FRESH tokens are appended
(the "margin", 10..500). Two paths per margin:

  * cached: fixed primed prefix + N fresh tokens  -> only the N tokens should be
            new work IF the prefix actually cached (block-granularity dependent).
  * cold:   fresh random prefix + N fresh tokens  -> guaranteed miss, full reprefill.

Verifies prefix-cache hit rate from /metrics deltas around the cached run.
Client needs only `requests`. Raw token-ids -> /v1/completions, 5 out-tok, stream,
TTFT = send -> first streamed chunk. Same timing method as ttft_sweep.py.
"""
import argparse, random, time, re
import requests

VOCAB_LO, VOCAB_HI = 100, 150_000


def fixed_prefix(n, seed=12345):
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def fresh(n):
    return random.choices(range(VOCAB_LO, VOCAB_HI), k=n)


def prime(url, model, ids):
    requests.post(url, json={"model": model, "prompt": ids, "max_tokens": 1,
                             "temperature": 0.0, "ignore_eos": True}).raise_for_status()


def ttft_once(url, model, ids, out_tokens=5):
    payload = {"model": model, "prompt": ids, "max_tokens": out_tokens,
               "stream": True, "temperature": 0.0, "ignore_eos": True}
    t0 = time.perf_counter()
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode() if isinstance(line, bytes) else line
            if s.startswith("data:"):
                d = s[5:].strip()
                if d and d != "[DONE]":
                    return time.perf_counter() - t0
    return time.perf_counter() - t0


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def cache_counters(metrics_url):
    """Return (hits, queries) summed over all label sets, or (None,None)."""
    try:
        txt = requests.get(metrics_url, timeout=5).text
    except Exception:
        return None, None
    def total(substr):
        tot = 0.0; seen = False
        for ln in txt.splitlines():
            if ln.startswith("#") or substr not in ln:
                continue
            try:
                tot += float(ln.rsplit(" ", 1)[1]); seen = True
            except ValueError:
                pass
        return tot if seen else None
    return total("prefix_cache_hits"), total("prefix_cache_queries")


def measure(url, model, build_ids, rounds, warm):
    samples = []
    for r in range(rounds):
        t = ttft_once(url, model, build_ids())
        if r >= warm:
            samples.append(t * 1000.0)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--metrics", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--prefix-len", type=int, default=4000)
    p.add_argument("--margins", default="0,10,50,100,200,300,400,500")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--warm", type=int, default=4)
    a = p.parse_args()

    margins = [int(x) for x in a.margins.split(",")]
    prefix = fixed_prefix(a.prefix_len)
    prime(a.url, a.model, prefix)   # land the prefix in cache before timing

    print(f"prefix_len={a.prefix_len}  rounds={a.rounds} warm={a.warm}")
    print(f"{'margin':>6} {'total':>6} {'cached_p50':>10} {'cached_min':>10} "
          f"{'cold_p50':>9} {'save_ms':>8} {'hit%':>6}")
    base = None
    for N in margins:
        prime(a.url, a.model, prefix)            # keep prefix warm
        h0, q0 = cache_counters(a.metrics)
        cached = measure(a.url, a.model, lambda: prefix + fresh(N), a.rounds, a.warm)
        h1, q1 = cache_counters(a.metrics)
        cold = measure(a.url, a.model, lambda: fresh(a.prefix_len) + fresh(N), a.rounds, a.warm)
        cp50, cmin = pctl(cached, .5), min(cached)
        dp50 = pctl(cold, .5)
        hit = ""
        if None not in (h0, q0, h1, q1) and (q1 - q0) > 0:
            hit = f"{100*(h1-h0)/(q1-q0):.0f}"
        if base is None:
            base = cp50
        print(f"{N:>6} {a.prefix_len+N:>6} {cp50:>10.1f} {cmin:>10.1f} "
              f"{dp50:>9.1f} {dp50-cp50:>8.1f} {hit:>6}", flush=True)


if __name__ == "__main__":
    main()
