# 2026-07-18 Day Log — Inkling-NVFP4 on 4× B200: C=1 kernel map (Nsight-calibrated)

Going into the kernels. Where does GPU time actually go for a single-stream (C=1)
prefill? First pass with the torch profiler (proportions only — its async-engine
capture window mangled absolute numbers), then **Nsight Systems with a
`cudaProfilerApi` capture range** for calibrated per-kernel durations. Companion to the
prefill/TTFT daylog (`2026-07-18-inkling-nvfp4-b200x4.md`).

**Headline (nsys, L=4096, eager):** ~**96 ms of real GPU compute per prefill, identical
across all 4 TP ranks**, split **MoE FP4 experts 37.5% / attention machinery 34.6% /
dense GEMM 23.7% / glue 4%**. But the forward is **~44% idle** — every rank spends
~74 ms not computing (launch gaps + a cross-rank sync); on 3 of 4 ranks that idle is
absorbed into a single spin-waiting kernel, `_reduce_insert_kernel` (65–77 ms), while
the lead rank shows it as gaps. The attention share is dominated by Inkling's custom
**sconv + relative-pos + sink** plumbing, *not* the FlashAttention-4 math (~7 ms of it).

## Method
Dedicated server, `--enforce-eager` (clean per-kernel attribution, no graph replay).
Two profilers:

1. **torch** (`--profiler-config.profiler=torch …`) via `/start_profile`+`/stop_profile`
   — good for kernel *identity* and *proportions*, but the async engine loop +
   `active_iterations=5` schedule made absolute/scaling numbers unreliable (a 2048 vs
   4096 capture gave a non-physical 6.5× gap). Use for names, not magnitudes.
2. **Nsight Systems** (authoritative magnitudes) — server run under nsys with capture
   keyed to `cudaProfilerStart` (vLLM `profiler=cuda` mode calls it on `/start_profile`),
   so nsys records *only* the single profiled prefill:
   ```bash
   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
     --trace=cuda,nvtx --cuda-graph-trace=node --sample=none -o /workspace/nsys/prefill4096 \
     vllm serve thinkingmachines/Inkling-NVFP4 --trust-remote-code --tokenizer-mode inkling \
       --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
       --tensor-parallel-size 4 --kernel-config.enable_flashinfer_autotune=False \
       --max-model-len 8192 --kv-cache-memory=8589934592 --enforce-eager \
       --profiler-config.profiler=cuda --host 0.0.0.0 --port 8000
   # driver: 5 warm prefills -> POST /start_profile -> 1 prefill (4096 rand ids, max_tokens=1) -> POST /stop_profile
   nsys stats --report cuda_gpu_kern_sum prefill4096.nsys-rep      # per-kernel GPU time
   # + per-GPU split via the .sqlite (CUPTI_ACTIVITY_KIND_KERNEL x StringIds)
   ```
   The `.nsys-rep` is only 709 KB — capture-range excluded all startup.

## Finding 1 — ~96 ms compute/GPU + ~44% idle; the idle hides in one sync kernel
Per-GPU, real nsys kernel durations (L=4096 prefill):
```
 GPU   total_ms   _reduce_insert_ms   compute_ms(excl sync)
   0     161.5           65.0                 96.5
   1     172.8           76.6                 96.2
   2     172.7           76.3                 96.4
   3      98.4            2.7                 95.7   <- lead rank this run
```
**Compute is ~96 ms on every rank** (the calibrated per-prefill number; torch-profiler
rank0 agreed at 98.6 ms). `_reduce_insert_kernel` is a per-layer cross-rank sync (132
launches/GPU = 2×/layer): **2.7 ms on the lead, 65–77 ms of spin-wait on the other 3**.
Same wall (~170 ms) on all ranks = ~96 ms compute + ~74 ms idle each; the lead shows
idle as inter-kernel gaps, the waiters as inflated `_reduce_insert`. So at C=1 the
forward is **~56% compute / ~44% idle** — the textbook latency/launch/sync-bound
signature, and exactly the idle that FULL CUDA-graphs (prefill daylog T4) and batching
(T3/EC3) recover. Note compute is *balanced* across ranks, so the sync bubble is
launch-gap + sync latency, not EP token-routing load imbalance.

## Finding 2 — the compute map (nsys, per GPU, L=4096)
~96.2 ms compute/GPU, excluding the `_reduce_insert` sync:
```
component                            ms/GPU   %compute   launches/GPU
MoE (FP4 experts + route)             36.1     37.5%        443
attention (FA4 + sconv/rel/sink)      33.3     34.6%        922
dense GEMM (qkv/o/shared/router)      22.8     23.7%        329
elementwise/glue                       3.9      4.1%        240
norm                                   0.0      0.0%          2
```
Top kernels by real GPU time (per-instance avg, aggregate over 4 GPUs):
```
  kernel                                  inst   avg_ns    what it is
  bmm_E2m1_E2m1E2m1_Fp32_...               252   247,207   MoE expert GEMM, FP4 (E2M1) tensor-core
  nvjet_sm100_tst_128x256_..._h            784    72,289   dense GEMM (cuBLASLt sm100): qkv/o/shared
  bmm_Bfloat16_E2m1E2m1_...                252   208,832   MoE expert GEMM, bf16×FP4
  _sconv_publish_kernel                    528    69,667   Inkling attn: short-conv publish
  _publish_input_kernel                    528    63,982   attn data movement
  nvjet_sm100_tst_128x256_..._v            228   108,498   dense GEMM
  moe::dev::finalize::finalizeKernel...    252    74,130   MoE combine expert outputs
  fa4flash_fwd_sm100...                    220    54,764   FlashAttention-4 forward (the actual attn)
  _gather_norm_kernel                      528    36,519   attn gather + norm
  fa4shearing_biasShearingBias...          220    39,242   FA4 shearing (rel-pos)
  _inkling_gate_select_kernel              256    19,351   MoE routing (top-6 of 256)
  _rel_proj_throughput_kernel              264    17,171   relative-position projection (d_rel=16)
  _kv_kernel / _sink_epilogue_kernel       264    ~15,400  KV proj / attention sink
  cvt_fp16_to_fp4_sf_major                 252     9,619   activation → FP4 quant for MoE
  routingIndicesClusterKernel              252     9,549   MoE expert routing
```

## Reading the architecture off the kernels
- **MoE experts run on native FP4 tensor cores** — the two `bmm_*E2m1*` kernels (E2M1 =
  FP4) are the largest compute item: (247+209 µs) × 63/GPU ≈ **28.7 ms/GPU**, confirming
  the `FLASHINFER_TRTLLM` backend uses the Blackwell FP4 path (no bf16 upconvert).
  +finalize/routing/gate ≈ 7 ms. ~37.5% of compute for top-6-of-256 + 2 shared experts.
- **Attention ≠ FlashAttention.** FA4 math (`fa4flash`+`fa4shearing`+`fa4cu_blocks`) is
  only ~7 ms/GPU; the other ~26 ms is Inkling's own machinery — `_sconv_*` (short conv
  over the sequence), `_rel_proj` (relative position, `d_rel=16`, `rel_extent=1024`),
  `_sink_epilogue` (attention sink; `shared_expert_sink` in config), plus
  `_publish`/`_gather` data movement. Many tiny kernels at **132 launches/GPU = 2×/layer**
  — movement/launch-bound. This is the model's distinctive part and the biggest bag of
  small kernels driving the launch-gap idle.
- **Dense GEMMs are cuBLASLt `nvjet` sm100** (qkv/o_proj/shared-expert/router), ~23 ms.
- **RMSNorm ~0** (fused via `fuse_norm_quant`); TP all-reduce doesn't appear as a named
  NCCL kernel — its cost is inside the `_reduce_insert` sync.

## What changed vs the torch-profiler pass
Nsight **confirms the proportions** (MoE 37 / attn 35 / GEMM 24) and **calibrates the
magnitudes** the torch profiler couldn't: ~96 ms compute/GPU (not the 15-vs-99 ms mess
the async window produced), and it pins the `_reduce_insert` sync at 65–77 ms of real
spin-wait on 3 ranks. The one open item from the previous daylog — "absolute magnitude
unresolved" — is now resolved.

## Open (next)
- **Length scaling done right:** repeat the nsys capture at L∈{1024,2048,6144} to get
  real per-token cost per bucket (MoE/GEMM ∝ tokens, attention movement partly fixed).
- **The ~44% idle:** re-profile in the *graphed* config (not eager) under nsys to
  measure how much of the 74 ms idle the CUDA graphs actually remove, and what sync
  residual remains — ties directly to the T4 result.
- **Decode step** if it ever matters (out of scope here).

```
Box: 4× B200 183 GiB, vLLM 0.1.dev18898+g93d5b2187 (inkling), NVFP4, TP=4 + EP,
--enforce-eager. Nsight Systems 2024.6.2, cudaProfilerApi capture range.
Scripts: prof_prefill.py, parse_trace.py, launch_nsys.sh.
```
