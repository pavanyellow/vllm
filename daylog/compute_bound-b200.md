# 2026-06-01 (B200) — moving to B200 is the whole win; the cold floor is now compute/kernel-bound

Carried the H100 cudagraph result (`2026-06-01-h100-cudagraph.md`) onto a **B200 (183 GiB,
compute cap 10.0, vLLM 0.22.0, CUDA 13)**. The H100 conclusion named two B200 levers: (1) the
~2.4× HBM3e bandwidth drops the 32 GB expert-read floor, and (2) the 183 GiB unblocks the G7
"graph the big cold chunk" experiment that OOM'd on 80 GB. Tested both. Lever 1 is the entire
story (−35…−41% across the board, *identical config*). Lever 2 is **partially falsified**: on
B200 the big-chunk graph buys ~2–8 ms and does **not** collapse the sawtooth — and the profiler
shows why: the cold prefill is GPU-bound, and the bottleneck has shifted *off* bandwidth onto
the gated-deltanet scan + dense GEMMs + quant/norm overhead.

Same model (`Qwen/Qwen3.6-35B-A3B-FP8`), same scripts (`ttft_sweep.py`, `kv_suffix.py`,
`prof_prefill.py`), same methodology as the H100 logs (C=1, random token-ids = zero cache,
TTFT p50 in ms). MoE backend on Blackwell auto-selects **CUTLASS FP8** (DeepGEMM auto-disabled:
"E8M0 scale format causes accuracy degradation for qwen3_5_moe") — a real sm100 FP8 tensor-core
path, not the Marlin-upconvert trap that sank the NVFP4 attempt (`2026-05-30-b200-fp4.md`).

## B1 — pure hardware swap: H100 G6 config, verbatim, on B200 (the whole win)
Same `vllm serve` as H100 G6 (graphs→1200, util 0.85). KV cache **3.36M tokens** (vs ~922k on
H100, 3.6×); graph pool 0.69 GiB.

| Path | H100 (G6, graphs→1200) | **B200 (same config)** | Δ |
|---|--:|--:|--:|
| Small graphed floor (L≤128) | ~22 | **17.7** | −20% |
| Cold L=500 | 34.7 | **22.4** | −35% |
| **Warm cached re-serve** (N≤500, 79% hit) | ~48 | **28.3** | **−41%** |
| **Cold 4k** (fresh, 0 reuse) | 109.4 | **72.7** | **−33%** |
| Cold 3168 (aligned, 3×1056) | 79.9 | **52.9** | −34% |
| Cold 4224 (aligned, 4×1056) | 93.8 | **62.0** | −34% |
| Decode (ms/tok) | 4.15 | **3.72** | −10% |

Warm re-serve is **flat 28.3 ms across N=0…500** (block-granular reuse + the graphed ~832-tok
reprefill tail; B200 reads the experts faster). Decode barely moves: at C=1 it's a latency-bound
40-layer chain (per-layer launch + float32 mamba recurrence), not an expert-bandwidth problem
(1 token routes to 8 experts ≈ 1 GB read, ~0.13 ms) — nothing for 2.4× HBM to bite on.

## B2 — G7 finally testable: graph the big cold chunk {2112, 3168, 4224}
The cold prefill for length L at block=1056 splits into a 1056-multiple **big chunk** + a ≤1056
remainder (e.g. 4000 → [3168 + 832]). Remainders were already graphed (≤1200 ladder); the big
chunk ran eager. Added rungs `1056×{2,3,4} = {2112,3168,4224}` — exactly the set the H100 G7
predicted would collapse the sawtooth, and exactly what OOM'd on 80 GB.

**It fits on B200, cheaply.** Persistent graph memory **1.36 GiB total** (+0.67 over the →1200
ladder). The H100 "each large capture ≈ 6–8.5 GiB" was the *transient* capture-workspace (freed
after capture), not persistent graph memory — that transient is what OOM'd. At util 0.85 the
4224-rung's ~8.5 GiB transient fits. **Ceiling: 4224.** Adding 5280/6336/7392 (for the full 0–8k
cold path) OOMs — the 7392 capture needs a **14.9 GiB transient** that doesn't fit once KV has
taken ~116 GiB. Since 4224 covers all cold ≤5k, that's the shippable ladder.

Cold 3k–5k, big-chunk graphs on vs off:

| L | graphs→1200 | graphs→4224 | Δ | | L | graphs→1200 | graphs→4224 | Δ |
|--:|--:|--:|--:|--|--:|--:|--:|--:|
| 3000 | 68.9 | 60.9 | −8.0 | | 4096 | 73.8 | 72.7 | −1.1 |
| 3168 *(aligned)* | 52.9 | 51.8 | −1.1 | | 4224 *(aligned)* | 62.0 | 60.8 | −1.2 |
| 3400 | 67.3 | 64.1 | −3.2 | | 4400 | 76.6 | 73.8 | −2.8 |
| 3600 | 70.2 | 67.2 | −3.0 | | 4600 | 78.3 | 76.6 | −1.7 |
| 3800 | 72.2 | 70.0 | −2.2 | | 4800 | 80.4 | 79.2 | −1.2 |
| 4000 | 74.5 | 71.3 | −3.2 | | 5000 | 82.0 | 82.0 | −1.5 |

**Sawtooth survives**: 4096/4224 = 72.7/60.8 = **1.20×** (was 1.19× without the big-chunk graph).
The G7 prediction — "graph the big chunk → cold band collapses to the aligned ~90 ms band" — is
**partially falsified on real hardware**. Graphing helps ~2–8 ms, not ~35 ms. Warm, small (≤500),
and decode are unchanged by the big rungs, as expected.

## Why graphs stopped helping — the profiler (this is the payoff)
`prof_prefill.py` on B200, single isolated prefill, device-kernel time bucketed (the FP8 MoE
GEMM shows up as `bmm_*dsFp8` and the deltanet as `chunk_*`/`merge_*` kernels — re-bucketed by
hand below):

**L=4224 (1-pass, aligned): 51.9 ms GPU-busy, 1726 kernels** (TTFT 60.8 ms ⇒ only ~9 ms is
non-GPU overhead — it's **GPU-bound**, no idle launch gap left for a graph to remove):

| Component | ms | % | bound by |
|---|--:|--:|---|
| Gated-deltanet scan (lin_attn + chunk/merge/recompute, 30 layers) | ~14.1 | 27% | compute / kernel overhead |
| MoE expert GEMM (`bmm_*dsFp8`, ×40) | ~13.1 | 25% | **bandwidth** (32 GB read, ~2.5 TB/s eff) |
| Dense GEMMs (QKV / o_proj / router / shared expert) | ~12.6 | 24% | mixed |
| Quant + norm + route | ~9.9 | 19% | overhead |
| Full attention (FMHA, 10 layers) | 1.3 | 2.6% | — |

Two things this kills:
- **The bandwidth read is only ~25% of the cold floor.** That's why B200 gave −35%, not the
  −58% the 2.4× HBM ratio would imply — only a quarter of the time rides the faster memory. The
  expert read dropped H100→B200 ~23 → ~13 ms; the rest didn't scale with bandwidth.
- **O(N²) attention is a non-issue (1.3 ms).** The earlier hypothesis that growing attention
  drives the length-slope was wrong. The slope (3168→52 vs 4224→61) is deltanet + dense GEMM +
  per-token quant/norm, not softmax.

**Sawtooth = a second full forward pass.** L=4000 (2-pass [3168+832]): **62.9 ms GPU-busy, 3460
kernels** — kernel count ~doubles and GPU-busy is +11 ms vs the 1-pass 4224, despite *fewer*
tokens (4000 < 4224). The non-aligned prompt re-runs the whole stack (second 32 GB expert read +
second deltanet + second quant/norm). That +11 ms GPU-busy ≈ the +10.5 ms TTFT sawtooth. It's a
second bandwidth+compute pass, with no exposed dispatch — graphs are structurally unable to touch
it. Killing it still needs 1-pass (block=8192), which still kills caching — same catch-22 as H100,
now at ~30% lower absolute latency.

### Shippable B200 config (validated)
`block=1056 + graphs→4224 + prefix caching` (util 0.85), the G6 command with the ladder extended
to `…,1152,1200,2112,3168,4224`:
- ✅ warm cached re-serve **~28 ms** (−41% vs H100), flat across N≤500
- ✅ small/mid ≤500 **17.7–22.4 ms** (−20…−35%)
- ✅ cold 4k **~71 ms** (−35% vs H100's 109) — the H100 "cold-90 needs H200" target is beaten
- ✅ decode **3.7 ms/tok**, outputs correct (`Paris`, etc.)
- 🔶 sawtooth **1.2×** (bandwidth, not dispatch — graphs don't help; 1-pass needs block=8192)

### Conclusion: B200 moved the wall from bandwidth to kernels
On H100 the story was "graphs delete the dispatch floor, then the MoE weight-read is the wall."
On B200 the dispatch floor was already gone *and* the weight-read is 2.4× cheaper, so the cold
prefill is now **GPU-bound on the gated-deltanet scan + dense GEMMs + quant/norm overhead**
(~70% combined) with the expert read down to ~25%. The remaining levers are **not** graphs and
**not** more bandwidth — they're kernel efficiency:
- **Gated-deltanet scan (~14 ms, 27%)** — the `chunk_*`/`merge_16x16_to_64x64_inverse` kernels;
  the H100 log already flagged this as ~140× the cost of a linear op. Largest single slice now.
- **Quant + norm + route (~10 ms, 19%)** — the "unoptimized overhead" the H100 roofline named;
  candidate for fusion (the build already enables `norm_quant`/`act_quant` fusions — more to do).
- **MoE GEMM efficiency (~13 ms at ~2.5 TB/s)** — bandwidth-bound but only ~30% of peak HBM;
  better grouped-GEMM tiling for the small per-expert M (~130 tok/expert) would recover some.

W4/INT4 experts would still halve the ~13 ms read, but it's now only a quarter of the floor —
diminishing returns at C=1. The deltanet scan is the new headline.
