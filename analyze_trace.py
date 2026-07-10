#!/usr/bin/env python3
"""Summarize a vLLM torch-profiler chrome trace (one rank).

Buckets GPU kernel time by component, reports wall span, GPU busy, idle,
and the top-N kernels by total time. Usage:
    python3 analyze_trace.py trace.json[.gz] [--top 30]
"""
import argparse, gzip, json, re, sys
from collections import defaultdict

BUCKETS = [  # (bucket, regex) — first match wins, order matters
    ("comms",       r"nccl|allreduce|all_reduce|cross_device_reduce|one_shot|two_shot|symm_mem|allgather|all_gather|reduce_scatter"),
    ("dsa_indexer", r"indexer|fp8_paged_mqa|mqa_logits|top_k_per_row|topk_indices|sparse_attn|lightning"),
    ("attention",   r"mla|fmha|flash|attention|attn|paged|softmax_stats|splitkv"),
    ("moe_gemm",    r"moe|grouped_gemm|group_gemm|expert|bmm_fp8|block_scale_moe"),
    ("dense_gemm",  r"gemm|cutlass|nvjet|matmul|mm_|sm100|wgmma|cublas"),
    ("quant_cvt",   r"quant|cvt_|convert|scaled_fp8|per_token_group|fp8_quant"),
    ("norm_act",    r"norm|silu|gelu|act_and_mul|activation|residual"),
    ("route_topk",  r"topk|top_k|route|sigmoid|grouped_top|select_expert|count_and_sort|expand|finalize"),
    ("copy_misc",   r"memcpy|copy|elementwise|cat_|concat|fill|arange|index|gather|scatter|embed|triton_poi|triton_per|triton_red"),
]

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    ev = load(a.trace)["traceEvents"]
    # GPU kernels: cat 'kernel' (torch profiler) on a GPU stream
    ks = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy") and e.get("dur", 0) > 0]
    if not ks:
        sys.exit("no kernel events found")
    t0 = min(e["ts"] for e in ks); t1 = max(e["ts"] + e["dur"] for e in ks)
    span_ms = (t1 - t0) / 1000.0
    # GPU busy: union of kernel intervals across all streams (merge overlaps)
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ks)
    busy = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs; busy_ms = busy / 1000.0
    bucket_t = defaultdict(float); kern_t = defaultdict(float); kern_n = defaultdict(int)
    for e in ks:
        n = e["name"]; nl = n.lower()
        kern_t[n] += e["dur"]; kern_n[n] += 1
        for b, rx in BUCKETS:
            if re.search(rx, nl): bucket_t[b] += e["dur"]; break
        else: bucket_t["OTHER"] += e["dur"]
    tot = sum(bucket_t.values()) / 1000.0
    print(f"span {span_ms:.1f} ms | gpu-busy {busy_ms:.1f} ms | idle {span_ms-busy_ms:.1f} ms ({100*(span_ms-busy_ms)/span_ms:.0f}%)")
    print(f"\nΣ kernel time by bucket (ms, % of Σ={tot:.1f}):")
    for b, t in sorted(bucket_t.items(), key=lambda x: -x[1]):
        print(f"  {b:<12} {t/1000:8.2f}  {100*t/1000/tot:5.1f}%")
    print(f"\ntop {a.top} kernels by Σ time:")
    for n, t in sorted(kern_t.items(), key=lambda x: -x[1])[:a.top]:
        print(f"  {t/1000:8.2f} ms  ×{kern_n[n]:<5} {n[:110]}")

if __name__ == "__main__":
    main()
