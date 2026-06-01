#!/usr/bin/env python3
"""Profile ONE prefix-cache-reuse prefill: a 4k prompt with the last N changed.

Primes a fixed `orig` (len L), warms the reuse request, then /start_profile -> one
`orig[:L-N] + fresh(N)` request (max_tokens=1) -> /stop_profile. Buckets device-kernel
dur from the worker trace (same regex as prof_prefill.py). Server must be launched
with --profiler-config. Compare against prof_prefill.py --len L (full cold prefill).
"""
import argparse, glob, gzip, json, os, random, re, time
import requests

VOCAB_LO, VOCAB_HI = 100, 150_000
BUCKETS = [
    ("lin_attn_gdn", r"deltarule|delta_rule|causal_conv|conv1d|post_conv|fused_recurrent|fused_chunk|chunk_scan|chunk_o|\bgdn\b|\bwy\b"),
    ("full_attn_fmha", r"flashattn|flash::|flash_fwd|fmha"),
    ("fp8_dynamic_quant", r"quant"),
    ("moe_grouped_gemm", r"fused_moe|grouped_gemm|m_grouped|moe_gemm"),
    ("moe_route_act", r"topkgating|topk|moe_align|moe_sum|count_and_sort|expert_token|act_and_mul|finalize|scatter|permute"),
    ("elementwise_norm", r"rms_norm|layernorm|\bnorm\b|elementwise|vectorized|reduce_kernel|rsqrt|mean_mul_pow|memcpy|memset|reshape_and_cache|index_|fill|exp_kernel|sigmoid|bitwise|to_copy|copy|\badd\b|silu|\bmul\b"),
    ("dense_gemm", r"sm90_fp8_gemm|deep_gemm|fp8_gemm|nvjet|cutlass|\bgemm\b|matmul"),
]


def fixed(n, seed=12345):
    rng = random.Random(seed); return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]
def fresh(n): return random.choices(range(VOCAB_LO, VOCAB_HI), k=n)
def send(url, model, ids, mt=1):
    requests.post(url, json={"model": model, "prompt": ids, "max_tokens": mt,
                  "temperature": 0.0, "ignore_eos": True}).raise_for_status()
def newest(d):
    fs = glob.glob(os.path.join(d, "**", "*.json*"), recursive=True)
    return max(fs, key=os.path.getmtime) if fs else None
def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f: return json.load(f)
def bkt(nm):
    for lb, rx in BUCKETS:
        if re.search(rx, nm, re.I): return lb
    return "OTHER"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--ctrl", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--len", type=int, default=4000)
    p.add_argument("--changed", type=int, default=50)
    p.add_argument("--warm", type=int, default=4)
    p.add_argument("--profile-dir", default="/vllm-workspace/prof_kv")
    a = p.parse_args()
    L, N = a.len, a.changed
    orig = fixed(L)
    send(a.url, a.model, orig)                       # prime full prefix
    before = set(glob.glob(os.path.join(a.profile_dir, "**", "*.json*"), recursive=True))
    for _ in range(a.warm):
        send(a.url, a.model, orig[:L - N] + fresh(N))  # warm the reuse shape
    requests.post(f"{a.ctrl}/start_profile").raise_for_status()
    send(a.url, a.model, orig[:L - N] + fresh(N))
    requests.post(f"{a.ctrl}/stop_profile").raise_for_status()
    path, last = None, -1
    for _ in range(120):
        time.sleep(1)
        new = set(glob.glob(os.path.join(a.profile_dir, "**", "*.json*"), recursive=True)) - before
        if new:
            path = max(new, key=os.path.getmtime); sz = os.path.getsize(path)
            if sz == last and sz > 0: break
            last = sz
    tr = load(path); evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in evs if e.get("ph") == "X" and "dur" in e and (e.get("cat") or "").lower() in ("kernel","gpu_memcpy","gpu_memset")]
    tot = sum(e["dur"] for e in ks)
    print(f"reuse prefill L={L} changed_last={N} (shared {L-N})  trace={os.path.basename(path)}")
    print(f"device kernels: {len(ks)}   GPU busy: {tot/1000:.2f} ms")
    buck = {}
    for e in ks:
        b = buck.setdefault(bkt(e.get("name","?")), [0.0, 0]); b[0] += e["dur"]; b[1] += 1
    for b, (d, c) in sorted(buck.items(), key=lambda x: -x[1][0]):
        print(f"  {d/1000:7.2f} ms ({100*d/tot:4.1f}%) x{c:<5} {b}")


if __name__ == "__main__":
    main()
