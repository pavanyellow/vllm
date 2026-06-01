#!/usr/bin/env python3
"""Latency of re-serving a 4k prompt with only the LAST N tokens changed.

Scenario: a 4000-token prompt was already served (its KV is in the prefix cache).
A new request arrives that shares the first (4000-N) tokens and changes the last N.
How much must be recomputed, and what's the TTFT? Prefix-cache reuse is BLOCK-
granular, so the answer depends on --block-size, not N directly.

  new_prompt = original[:L-N] + fresh(N)

N=0 = identical request (max cache hit). Baseline 'cold' = a fully fresh L-token
prompt (0 reuse). Verifies hit rate from /metrics deltas. Needs only `requests`.
"""
import argparse, random, time
import requests

VOCAB_LO, VOCAB_HI = 100, 150_000


def fixed(n, seed=12345):
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def fresh(n):
    return random.choices(range(VOCAB_LO, VOCAB_HI), k=n)


def prime(url, model, ids):
    requests.post(url, json={"model": model, "prompt": ids, "max_tokens": 1,
                             "temperature": 0.0, "ignore_eos": True}).raise_for_status()


def ttft_once(url, model, ids):
    pl = {"model": model, "prompt": ids, "max_tokens": 5, "stream": True,
          "temperature": 0.0, "ignore_eos": True}
    t0 = time.perf_counter()
    with requests.post(url, json=pl, stream=True) as r:
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
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def counters(murl):
    try:
        txt = requests.get(murl, timeout=5).text
    except Exception:
        return None, None
    def tot(sub):
        s = 0.0; seen = False
        for ln in txt.splitlines():
            if ln.startswith("#") or sub not in ln:
                continue
            try: s += float(ln.rsplit(" ", 1)[1]); seen = True
            except ValueError: pass
        return s if seen else None
    return tot("prefix_cache_hits"), tot("prefix_cache_queries")


def run(url, model, build, rounds, warm):
    out = []
    for r in range(rounds):
        t = ttft_once(url, model, build())
        if r >= warm: out.append(t * 1000)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--metrics", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--len", type=int, default=4000)
    p.add_argument("--changed", default="0,10,20,50,100,500")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--warm", type=int, default=4)
    a = p.parse_args()
    L = a.len
    orig = fixed(L)
    prime(a.url, a.model, orig)

    cold = run(a.url, a.model, lambda: fresh(L), a.rounds, a.warm)  # 0-reuse baseline
    print(f"len={L}  rounds={a.rounds} warm={a.warm}")
    print(f"full-cold (fresh {L}, 0 reuse): p50={pctl(cold,.5):.1f} min={min(cold):.1f} ms")
    print(f"{'changed':>7} {'shared':>6} {'TTFT_p50':>8} {'TTFT_min':>8} {'hit%':>6} {'vs_cold':>7}")
    cbase = pctl(cold, .5)
    for N in [int(x) for x in a.changed.split(",")]:
        prime(a.url, a.model, orig)
        h0, q0 = counters(a.metrics)
        s = run(a.url, a.model, lambda: orig[:L - N] + fresh(N), a.rounds, a.warm)
        h1, q1 = counters(a.metrics)
        hit = f"{100*(h1-h0)/(q1-q0):.0f}" if None not in (h0,q0,h1,q1) and q1>q0 else ""
        print(f"{N:>7} {L-N:>6} {pctl(s,.5):>8.1f} {min(s):>8.1f} {hit:>6} "
              f"{cbase-pctl(s,.5):>7.1f}", flush=True)


if __name__ == "__main__":
    main()
