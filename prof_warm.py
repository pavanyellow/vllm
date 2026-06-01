#!/usr/bin/env python3
"""Profile the WARM cache-hit reprefill: a 4000-tok prompt is primed (cached), then a
request that shares the first L-N tokens and changes the last N is profiled. With
block=1056, cache reuse is block-granular, so only the trailing partial block(s) are
recomputed. Reuses prof_prefill.py's bucketing."""
import argparse, glob, gzip, json, os, random, re, time, requests
VLO, VHI = 100, 150_000
import sys; sys.path.insert(0, os.path.dirname(__file__))
from prof_prefill import BUCKETS, is_device_kernel, bucketize, newest_trace, load_trace

def fixed(n, seed=12345):
    rng = random.Random(seed); return [rng.randrange(VLO, VHI) for _ in range(n)]
def fresh(n): return random.choices(range(VLO, VHI), k=n)
def send(url, model, ids, mt=1):
    requests.post(url, json={"model": model, "prompt": ids, "max_tokens": mt,
                  "temperature": 0.0, "ignore_eos": True}).raise_for_status()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--ctrl", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--len", type=int, default=4000)
    p.add_argument("--changed", type=int, default=200)
    p.add_argument("--warm", type=int, default=4)
    p.add_argument("--profile-dir", default="/vllm-workspace/prof")
    a = p.parse_args()
    orig = fixed(a.len)
    before = set(glob.glob(os.path.join(a.profile_dir, "**", "*.json*"), recursive=True))
    send(a.url, a.model, orig)                       # prime the 4k prefix (cache it)
    for _ in range(a.warm):                           # warm the suffix shape
        send(a.url, a.model, orig[:a.len - a.changed] + fresh(a.changed))
    send(a.url, a.model, orig)                         # re-prime (warm reqs evicted nothing; ensure hit)
    requests.post(f"{a.ctrl}/start_profile").raise_for_status()
    send(a.url, a.model, orig[:a.len - a.changed] + fresh(a.changed))
    requests.post(f"{a.ctrl}/stop_profile").raise_for_status()
    path, last = None, -1
    for _ in range(120):
        time.sleep(1)
        new = set(glob.glob(os.path.join(a.profile_dir, "**", "*.json*"), recursive=True)) - before
        if new:
            path = max(new, key=os.path.getmtime); sz = os.path.getsize(path)
            if sz == last and sz > 0: break
            last = sz
    if not path: raise SystemExit("no new trace")
    tr = load_trace(path); evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kernels = [e for e in evs if is_device_kernel(e)]
    total = sum(e["dur"] for e in kernels)
    print(f"WARM len={a.len} changed={a.changed}  trace={os.path.basename(path)}")
    print(f"device kernels: {len(kernels)}  total GPU busy: {total/1000:.2f} ms")
    by_name = {}
    for e in kernels:
        d = by_name.setdefault(e.get("name", "?"), [0.0, 0]); d[0] += e["dur"]; d[1] += 1
    buck = {}
    for nm, (dur, cnt) in by_name.items():
        v = buck.setdefault(bucketize(nm), [0.0, 0]); v[0] += dur; v[1] += cnt
    print("--- buckets (ms) ---")
    for b, (dur, cnt) in sorted(buck.items(), key=lambda x: -x[1][0]):
        print(f"{dur/1000:8.3f} ms  ({100*dur/total:4.1f}%)  x{cnt:<6} {b}")
    print("--- top OTHER kernels ---")
    for nm, (dur, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0]):
        if bucketize(nm) == "OTHER":
            print(f"{dur/1000:8.3f} ms  x{cnt:<5} {nm[:90]}")

if __name__ == "__main__": main()
