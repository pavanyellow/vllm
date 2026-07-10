#!/usr/bin/env python3
"""Roofline model for GLM-5.2-FP8 prefill on 8xB200 (TP8), per-GPU.

Peaks used (B200, per GPU): FP8 dense ~4.5e15 FLOP/s, HBM3e ~8e12 B/s.
Prints expected memory-time and compute-time per component at length L,
to compare against measured bucket times (analyze_trace.py).
"""
import sys

H = 6144
MOE_I = 2048; N_EXP = 256; TOPK = 8; N_SHARED = 1
DENSE_I = 12288; N_DENSE = 3; N_LAYERS = 78; N_MOE = N_LAYERS - N_DENSE
HEADS = 64; QK_NOPE = 192; QK_ROPE = 64; V_DIM = 256
Q_LORA = 2048; KV_LORA = 512
IDX_HEADS = 32; IDX_DIM = 128; IDX_TOPK = 2048
PEAK_F = 4.5e15; PEAK_B = 8e12; NGPU = 8

def ms(x): return x * 1000

def table(L):
    rows = []
    # --- MoE routed experts (per GPU: 1/8 of every expert, all experts hit at L>=~1k)
    w_moe = N_EXP * 3 * H * MOE_I / NGPU            # bytes FP8 per layer per GPU
    f_moe = TOPK * 3 * 2 * H * MOE_I * L / NGPU     # FLOPs per layer per GPU
    rows.append(("MoE routed GEMM", N_MOE * w_moe / PEAK_B, N_MOE * f_moe / PEAK_F))
    # --- shared expert + dense-layer MLPs
    w_sh = 3 * H * MOE_I / NGPU; f_sh = 3 * 2 * H * MOE_I * L / NGPU
    w_d = 3 * H * DENSE_I / NGPU; f_d = 3 * 2 * H * DENSE_I * L / NGPU
    rows.append(("shared+dense MLP", (N_MOE * w_sh + N_DENSE * w_d) / PEAK_B,
                 (N_MOE * f_sh + N_DENSE * f_d) / PEAK_F))
    # --- MLA projections (dense GEMMs, TP8 over heads)
    w_proj = (H * Q_LORA + Q_LORA * HEADS * (QK_NOPE + QK_ROPE) + H * (KV_LORA + QK_ROPE)
              + KV_LORA * HEADS * (QK_NOPE + V_DIM) + HEADS * V_DIM * H) / NGPU
    rows.append(("MLA projections", N_LAYERS * w_proj / PEAK_B,
                 N_LAYERS * 2 * w_proj * L / PEAK_F))
    # --- fmha core: DSA sparse, ctx = min(pos, IDX_TOPK); qk dim 256, v 256
    ctx_sum = sum(min(p, IDX_TOPK) for p in range(L))
    f_fmha = 2 * ctx_sum * HEADS * ((QK_NOPE + QK_ROPE) + V_DIM) / NGPU * N_LAYERS
    kv_bytes = L * (KV_LORA + QK_ROPE) * N_LAYERS / NGPU  # fp8 latent KV read (approx, per GPU share)
    rows.append(("fmha core (DSA)", kv_bytes * min(L, IDX_TOPK) / max(L,1) / PEAK_B,
                 f_fmha / PEAK_F))
    # --- DSA indexer: logits over full ctx (quadratic), fp8
    ctx_full = L * (L + 1) / 2
    f_idx = 2 * ctx_full * IDX_HEADS * IDX_DIM / NGPU * N_LAYERS
    rows.append(("DSA indexer", 0.0, f_idx / PEAK_F))
    # --- allreduce: 2 per layer, L*H bf16 bytes each; NVLink multimem ~0.75*900GB/s eff
    b_ar = 2 * N_LAYERS * L * H * 2
    rows.append(("all-reduce (NVLink)", b_ar / (0.75 * 9e11), 0.0))
    print(f"\n=== L={L} — per-GPU roofline (ms) ===")
    print(f"{'component':<20} {'mem_roof':>9} {'comp_roof':>10} {'bound':>8}")
    tm = tc = 0.0
    for n, m, c in rows:
        b = "mem" if m > c else "comp"
        tm += m; tc += c
        print(f"{n:<20} {ms(m):9.2f} {ms(c):10.2f} {b:>8}")
    print(f"{'TOTAL (max per row)':<20} {ms(sum(max(m,c) for _,m,c in rows)):9.2f}")

for L in (int(x) for x in (sys.argv[1:] or [1000, 2000, 4000, 8000])):
    table(L)
