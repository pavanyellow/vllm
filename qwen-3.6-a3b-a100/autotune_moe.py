"""Autotune the Triton fused_moe_kernel for our specific shapes.

vLLM ships pre-tuned config JSONs for many (GPU × E × N × dtype) combos
but not for our combo (A100 PG509-210, E=256, N=512, INT8). We tune it
ourselves.

For each target token count, try many (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
num_warps, num_stages) combinations, time each, pick the best, write JSON
to the path vLLM looks at at load time:

  vllm/model_executor/layers/fused_moe/configs/
    E=<num_experts>,N=<intermediate>,device_name=<gpu>,dtype=int8_w8a8.json

This is what `vllm/benchmarks/kernels/benchmark_moe.py` does upstream; we
inline the logic since that script doesn't ship with pip install.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


# Wider sweep — Day-1 aggressive exploration over the BM/BN/BK/GM/warps/stages
# space. Prior 216-config sweep gave no improvement (PLAN.md); this adds
# BM=256, BK=256, ns=5, GM=64, nw=2 to cover edges that were missed.
# ~650 configs × ~3s JIT each ≈ 30 min per token count.
SWEEP_CONFIGS = []
for BM in (32, 64, 128, 256):              # M-block
    for BN in (64, 128, 256):              # N-block
        for BK in (64, 128, 256):          # K-block
            for GM in (1, 16, 64):         # group M
                for nw in (4, 8):          # num_warps
                    for ns in (3, 4, 5):   # num_stages
                        # Triton OutOfResources guard: shared mem ≈ BM*BN bytes
                        # for INT8 accumulator + register backing. A100 has 164KB
                        # shared mem / SM. Skip configs that won't fit.
                        if BM * BN > 32768:
                            continue
                        if BM * BK > 65536:
                            continue
                        SWEEP_CONFIGS.append({
                            "BLOCK_SIZE_M": BM,
                            "BLOCK_SIZE_N": BN,
                            "BLOCK_SIZE_K": BK,
                            "GROUP_SIZE_M": GM,
                            "num_warps": nw,
                            "num_stages": ns,
                        })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--shard-intermediate-size", type=int, default=512,
                   help="N dim of the (fc1+fc3 fused) expert matrix")
    p.add_argument("--dtype", choices=["int8_w8a8", "fp8_w8a8", "bf16"], default="int8_w8a8")
    p.add_argument("--token-counts", default="64,128,256,512,1024,2048,4096",
                   help="comma-separated token counts to tune for")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--max-configs", type=int, default=None,
                   help="limit configs swept per shape (None = all)")
    return p.parse_args()


def build_dummy_inputs(num_tokens: int, num_experts: int, top_k: int,
                       hidden_size: int, N: int, dtype: str):
    """Construct random MoE-shaped tensors for the fused kernel."""
    device = "cuda"

    if dtype == "int8_w8a8":
        a = torch.randint(-127, 127, (num_tokens, hidden_size), dtype=torch.int8, device=device)
        w1 = torch.randint(-127, 127, (num_experts, 2 * N, hidden_size), dtype=torch.int8, device=device)
        w2 = torch.randint(-127, 127, (num_experts, hidden_size, N), dtype=torch.int8, device=device)
    elif dtype == "fp8_w8a8":
        a = (torch.randn(num_tokens, hidden_size, device=device) * 4).to(torch.float8_e4m3fn)
        w1 = (torch.randn(num_experts, 2 * N, hidden_size, device=device) * 0.1).to(torch.float8_e4m3fn)
        w2 = (torch.randn(num_experts, hidden_size, N, device=device) * 0.1).to(torch.float8_e4m3fn)
    else:  # bf16
        a = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        w1 = torch.randn(num_experts, 2 * N, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
        w2 = torch.randn(num_experts, hidden_size, N, dtype=torch.bfloat16, device=device) * 0.1

    # router output: assign each token to top_k experts uniformly
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=torch.float32), dim=-1
    )
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)

    # scales for int8/fp8 paths
    if dtype in ("int8_w8a8", "fp8_w8a8"):
        a1_scale = torch.rand(num_tokens, dtype=torch.float32, device=device) + 0.01
        a2_scale = torch.rand(num_tokens, dtype=torch.float32, device=device) + 0.01
        w1_scale = torch.rand(num_experts, 2 * N, dtype=torch.float32, device=device) + 0.01
        w2_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.01
    else:
        a1_scale = a2_scale = w1_scale = w2_scale = None

    return a, w1, w2, topk_weights, topk_ids, a1_scale, a2_scale, w1_scale, w2_scale


def time_config(cfg: dict, inputs, use_int8_w8a8: bool, use_fp8_w8a8: bool,
                n_warmup: int, n_trials: int) -> float | None:
    """Run fused_experts with a forced config; return median ms or None on error."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        dispatch_fused_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    import triton.language as tl

    a, w1, w2, topk_weights, topk_ids, a1_scale, a2_scale, w1_scale, w2_scale = inputs

    try:
        # Build sorted token IDs (the routing arrangement the kernel expects)
        # This wraps moe_align_block_size which itself calls a small kernel.
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids, cfg["BLOCK_SIZE_M"], w1.size(0)
        )

        # Allocate intermediate output for fc1 — must be 3D (M, top_k, 2*N)
        # because the kernel indexes C.stride(2). See fused_experts_impl which
        # allocates: intermediate_cache1.view(M, top_k_num, N).
        E, N2, K = w1.shape  # N2 = 2*intermediate
        M = a.size(0)
        top_k = topk_ids.size(1)
        intermediate = torch.empty((M, top_k, N2), dtype=torch.bfloat16, device="cuda")

        def run_one():
            dispatch_fused_moe_kernel(
                A=a, B=w1, C=intermediate,
                A_scale=a1_scale, B_scale=w1_scale,
                B_zp=None, topk_weights=topk_weights,
                sorted_token_ids=sorted_ids, expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_pad,
                mul_routed_weight=False,
                top_k=topk_ids.size(1), config=cfg,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=True,
                block_shape=None,
            )

        # warmup
        for _ in range(n_warmup):
            run_one()
        torch.cuda.synchronize()

        # timed trials
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_one()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        return sorted(times)[len(times) // 2]
    except Exception as e:
        # uncomment to debug:
        if os.environ.get("AUTOTUNE_DEBUG"):
            import traceback
            print(f"[AUTOTUNE_ERR] cfg={cfg}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        return None


def main():
    args = parse_args()
    token_counts = [int(x) for x in args.token_counts.split(",")]

    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"

    configs = SWEEP_CONFIGS[:args.max_configs] if args.max_configs else SWEEP_CONFIGS
    print(f"sweeping {len(configs)} configs × {len(token_counts)} token counts")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: E={args.num_experts}, N={args.shard_intermediate_size}, hidden={args.hidden_size}")
    print()

    best_per_M = {}
    for M in token_counts:
        # build inputs once per M
        inputs = build_dummy_inputs(
            num_tokens=M, num_experts=args.num_experts, top_k=args.top_k,
            hidden_size=args.hidden_size, N=args.shard_intermediate_size,
            dtype=args.dtype,
        )
        print(f"=== M={M} ===")
        best_time = float("inf")
        best_cfg = None
        for i, cfg in enumerate(configs):
            t = time_config(cfg, inputs, use_int8_w8a8, use_fp8_w8a8,
                            args.n_warmup, args.n_trials)
            if t is not None and t < best_time:
                best_time = t
                best_cfg = cfg
            if (i + 1) % 50 == 0:
                print(f"  scanned {i+1}/{len(configs)}, best so far: {best_time:.2f}ms")
        if best_cfg is None:
            print(f"  NO valid config for M={M}")
            continue
        print(f"  BEST: {best_time:.2f}ms with {best_cfg}")
        best_per_M[M] = best_cfg

    # Write the JSON file vLLM expects
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_tag = f",dtype={args.dtype}" if args.dtype != "bf16" else ""
    fname = (f"E={args.num_experts},N={args.shard_intermediate_size},"
             f"device_name={gpu_name}{dtype_tag}.json")
    import vllm.model_executor.layers.fused_moe as fmoe_mod
    configs_dir = Path(fmoe_mod.__file__).parent / "configs"
    out_path = configs_dir / fname

    # Format vLLM expects: { "M_string": cfg, ... }
    out = {str(M): cfg for M, cfg in best_per_M.items()}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
