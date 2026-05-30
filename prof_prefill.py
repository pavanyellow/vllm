#!/usr/bin/env python3
"""Profile a single prefill pass and attribute device-kernel time by component.

Drives vLLM's torch profiler around ONE isolated prefill request (C=1,
max_tokens=1), then parses the worker chrome-trace and buckets every device
kernel (cat="kernel" / gpu_memcpy / gpu_memset) by a name regex.

Server must be launched with:
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>",
                      "torch_profiler_record_shapes":true,
                      "torch_profiler_with_flops":true}'

Procedure (matches B200 E7):
  1. Warm the shape with --warm prefill reqs (so compile/caches are hot).
  2. POST /start_profile -> 1 prefill req at length L -> POST /stop_profile.
  3. Find the newest trace file in the profiler dir, sum kernel `dur` by bucket.

Usage:
  python prof_prefill.py --len 4224 --profile-dir /vllm-workspace/prof
  python prof_prefill.py --len 4224 --list   # dump top kernels, no bucketing
"""
import argparse
import glob
import gzip
import json
import os
import random
import re
import time

import requests

VOCAB_LO, VOCAB_HI = 100, 150_000

# Buckets: (label, regex over kernel name, case-insensitive). First match wins,
# so order is most-specific -> generic. The gated-deltanet (DeltaRule) and
# FlashAttn kernels are both wrapped in `cutlass::device_kernel<...>`, so the
# generic gemm/cutlass rule MUST come last or it would steal them.
# Anything unmatched -> OTHER, printed individually so buckets can be refined.
BUCKETS = [
    ("lin_attn_gdn",        r"deltarule|delta_rule|causal_conv|conv1d|post_conv|fused_recurrent|fused_chunk|chunk_scan|chunk_o|\bgdn\b|\bwy\b"),
    ("full_attn_fmha",      r"flashattn|flash::|flash_fwd|fmha"),
    ("fp8_dynamic_quant",   r"quant"),
    ("moe_grouped_gemm",    r"fused_moe|grouped_gemm|m_grouped|moe_gemm"),
    ("moe_route_act",       r"topkgating|topk|moe_align|moe_sum|count_and_sort|expert_token|act_and_mul|finalize|scatter|permute"),
    ("elementwise_norm",    r"rms_norm|layernorm|\bnorm\b|elementwise|vectorized|reduce_kernel|rsqrt|mean_mul_pow|memcpy|memset|reshape_and_cache|index_|fill|exp_kernel|sigmoid|bitwise|to_copy|copy|\badd\b|silu|\bmul\b"),
    ("dense_gemm",          r"sm90_fp8_gemm|deep_gemm|fp8_gemm|nvjet|cutlass|\bgemm\b|matmul"),
]


def rand_ids(n):
    return [random.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def prefill(url, model, ids):
    r = requests.post(url, json={"model": model, "prompt": ids, "max_tokens": 1,
                                 "temperature": 0.0, "ignore_eos": True})
    r.raise_for_status()


def newest_trace(d):
    files = glob.glob(os.path.join(d, "**", "*.json*"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def is_device_kernel(ev):
    if ev.get("ph") != "X" or "dur" not in ev:
        return False
    cat = (ev.get("cat") or "").lower()
    return cat in ("kernel", "gpu_memcpy", "gpu_memset")


def bucketize(name):
    for label, rx in BUCKETS:
        if re.search(rx, name, re.IGNORECASE):
            return label
    return "OTHER"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--ctrl", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--len", type=int, default=4224)
    p.add_argument("--warm", type=int, default=4)
    p.add_argument("--profile-dir", default="/vllm-workspace/prof")
    p.add_argument("--list", action="store_true", help="dump top kernels, skip buckets")
    p.add_argument("--topn", type=int, default=40)
    args = p.parse_args()

    before = set(glob.glob(os.path.join(args.profile_dir, "**", "*.json*"), recursive=True))

    for _ in range(args.warm):
        prefill(args.url, args.model, rand_ids(args.len))

    requests.post(f"{args.ctrl}/start_profile").raise_for_status()
    prefill(args.url, args.model, rand_ids(args.len))
    requests.post(f"{args.ctrl}/stop_profile").raise_for_status()

    # Wait for a NEW trace file to be flushed and stable.
    path, last = None, -1
    for _ in range(120):
        time.sleep(1)
        cur = set(glob.glob(os.path.join(args.profile_dir, "**", "*.json*"), recursive=True))
        new = cur - before
        if new:
            path = max(new, key=os.path.getmtime)
            sz = os.path.getsize(path)
            if sz == last and sz > 0:
                break
            last = sz
    if not path:
        raise SystemExit("no new trace file appeared in %s" % args.profile_dir)
    print(f"trace: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    tr = load_trace(path)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kernels = [e for e in evs if is_device_kernel(e)]
    total = sum(e["dur"] for e in kernels)
    print(f"device kernels: {len(kernels)}  total GPU busy: {total/1000:.2f} ms")

    # per-name aggregation
    by_name = {}
    for e in kernels:
        nm = e.get("name", "?")
        d = by_name.setdefault(nm, [0.0, 0])
        d[0] += e["dur"]; d[1] += 1

    if args.list:
        print(f"\n--- top {args.topn} kernels by total dur ---")
        for nm, (dur, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:args.topn]:
            print(f"{dur/1000:8.3f} ms  x{cnt:<5} {nm[:110]}")
        return

    # bucketed
    buck = {}
    for nm, (dur, cnt) in by_name.items():
        b = bucketize(nm)
        v = buck.setdefault(b, [0.0, 0])
        v[0] += dur; v[1] += cnt
    print("\n--- buckets (ms) ---")
    for b, (dur, cnt) in sorted(buck.items(), key=lambda x: -x[1][0]):
        print(f"{dur/1000:8.3f} ms  ({100*dur/total:4.1f}%)  x{cnt:<6} {b}")
    if "OTHER" in buck:
        print("\n--- OTHER (unbucketed) kernels ---")
        for nm, (dur, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0]):
            if bucketize(nm) == "OTHER":
                print(f"{dur/1000:8.3f} ms  x{cnt:<5} {nm[:110]}")


if __name__ == "__main__":
    main()
