#!/usr/bin/env python3
"""Drive SGLang's torch profiler around one isolated request and bucket device
kernels by name — SGLang analogue of vLLM's prof_prefill.py (E10).

  prefill mode: L tokens, max_tokens=1  -> one prefill forward
  e2e mode:     L tokens, max_tokens=D  -> prefill + D-1 decode steps
"""
import argparse, glob, gzip, json, os, random, time
import requests

VLO, VHI = 100, 150_000
BASE = "http://127.0.0.1:30000"

# most-specific -> generic; first match wins. mirrors prof_prefill.py buckets.
BUCKETS = [
    ("gated-deltanet (lin_attn scan/conv)", r"chunk_|delta_rule|deltanet|gdn|causal_conv|fused_recurrent|fused_chunk|cumsum|l2norm|solve_tril|wy_|fwd_prepare|fwd_o|recompute"),
    ("MoE expert GEMM", r"fused_moe|grouped_gemm|group_gemm|moe_align|w8a8_block|silu_and_mul.*moe|deep_gemm|gemm_grouped"),
    ("full attn (flash)", r"flash|fmha|attention.*kernel|_attn_|mha_|fa3|fwd_kernel"),
    ("dense proj GEMM (qkv/o/router/shared)", r"cutlass.*gemm|gemm|cublas|s16816|wgmma|fp8.*gemm|nvjet|sm90.*gemm|ampere.*gemm|matmul"),
    ("FP8 quant", r"quant|scaled_fp8|per_token|cvt_|fp8_quant|to_fp8|dynamic_scaled"),
    ("rmsnorm/elementwise/rope", r"rms_norm|rmsnorm|layernorm|norm_|rope|rotary|add_|mul_|silu|gelu|act_|elementwise|index_|cat_|copy|vectorized"),
    ("reduce/topk/sort (routing)", r"topk|argmax|sort|reduce|softmax|arg_|cub::|scan|radix"),
]


def rand_ids(n):
    return [random.randrange(VLO, VHI) for _ in range(n)]


def gen(ids, max_tokens):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/completions", json={
        "model": "Qwen/Qwen3.6-35B-A3B", "prompt": ids, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": False})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, r.json().get("usage", {})


def newest_trace(d, before):
    files = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
             if f.endswith((".json", ".json.gz", ".trace.json", ".trace.json.gz"))
             and os.path.getmtime(f) >= before - 1]
    return max(files, key=os.path.getmtime) if files else None


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def bucket(trace):
    import re
    pats = [(lbl, re.compile(rx, re.I)) for lbl, rx in BUCKETS]
    sums = {lbl: 0.0 for lbl, _ in BUCKETS}
    other = {}
    total = 0.0
    evs = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    for e in evs:
        if not isinstance(e, dict):
            continue
        cat = (e.get("cat") or "").lower()
        if "kernel" not in cat and "gpu_memcpy" not in cat and "gpu_memset" not in cat:
            continue
        dur = e.get("dur", 0) or 0
        name = e.get("name", "")
        total += dur
        for lbl, rx in pats:
            if rx.search(name):
                sums[lbl] += dur
                break
        else:
            key = name[:48]
            other[key] = other.get(key, 0) + dur
    return sums, other, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=4200)
    ap.add_argument("--decode", type=int, default=1, help="max_tokens (1=prefill only)")
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--dir", default="/vllm-workspace/sgl_prof")
    ap.add_argument("--by-stage", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    ids = rand_ids(a.len)
    for _ in range(a.warm):
        gen(ids, a.decode)
    t_start = time.time()
    req = {"output_dir": a.dir, "activities": ["CPU", "GPU"], "record_shapes": True}
    if a.by_stage:
        req["profile_by_stage"] = True
    requests.post(f"{BASE}/start_profile", json=req).raise_for_status()
    ms, usage = gen(rand_ids(a.len), a.decode)
    requests.post(f"{BASE}/stop_profile").raise_for_status()
    time.sleep(3)  # let trace flush
    tr = newest_trace(a.dir, t_start)
    print(f"len={a.len} max_tokens={a.decode} wall={ms:.1f} ms usage={usage}")
    if not tr:
        print("NO TRACE FOUND in", a.dir); return
    print("trace:", tr)
    sums, other, total = bucket(load(tr))
    print(f"\n{'bucket':45} {'ms':>9}  {'%':>5}")
    busy = total / 1000.0
    for lbl, _ in BUCKETS:
        v = sums[lbl] / 1000.0
        print(f"{lbl:45} {v:9.3f}  {100*v/busy if busy else 0:5.1f}")
    oth = sum(other.values()) / 1000.0
    print(f"{'OTHER (unmatched)':45} {oth:9.3f}  {100*oth/busy if busy else 0:5.1f}")
    print(f"{'Σ GPU busy':45} {busy:9.3f}")
    top = sorted(other.items(), key=lambda kv: -kv[1])[:8]
    if top:
        print("\ntop unmatched kernels:")
        for k, v in top:
            print(f"  {v/1000.0:8.3f} ms  {k}")


if __name__ == "__main__":
    main()
