# 2026-07-18 Day Log — Inkling-NVFP4 on 4× B200: C=1 kernel map

Going into the kernels. Where does GPU time actually go for a single-stream (C=1)
prefill? Torch-profiler traces, per-rank, leaf GPU kernels only. Companion to the
prefill/TTFT daylog (`2026-07-18-inkling-nvfp4-b200x4.md`).

**Headline:** at C=1 the forward is ~half idle — 3 of 4 TP ranks park ~half the window
in a single cross-rank sync kernel (`_reduce_insert_kernel`), and rank0 shows the same
idle as inter-kernel launch gaps. Of the *compute* that does run, it splits roughly
**MoE FP4 experts ≈ 37% / attention machinery ≈ 35% / dense GEMM ≈ 25%** — and the
"attention" share is dominated by Inkling's custom **sconv + relative-pos + sink**
plumbing, *not* the FlashAttention-4 math (which is only ~7 ms of it).

## Profiling setup
Dedicated server, **`--enforce-eager`** (clean per-kernel attribution, no graph-replay
opacity) + torch profiler:
```bash
vllm serve thinkingmachines/Inkling-NVFP4 \
  --trust-remote-code --tokenizer-mode inkling --reasoning-parser inkling \
  --tool-call-parser inkling --enable-auto-tool-choice \
  --tensor-parallel-size 4 --kernel-config.enable_flashinfer_autotune=False \
  --max-model-len 8192 --kv-cache-memory=8589934592 --enforce-eager \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/workspace/prof \
  --profiler-config.torch_profiler_record_shapes=true \
  --profiler-config.torch_profiler_with_flops=true \
  --profiler-config.torch_profiler_with_stack=false \
  --host 0.0.0.0 --port 8000
```
Driver (`prof_prefill.py`): 5 warm prefills → `POST /start_profile` → **1 prefill**
(random token-ids, `max_tokens=1`) → `POST /stop_profile`. Parser (`parse_trace.py`)
sums `cat=="kernel"` device events per name and buckets by an Inkling-aware classifier.
Traces are per rank: `dp0_pp0_tp{r}_..._rank{r}.pt.trace.json.gz`.

## Finding 1 — C=1 is ~50% idle, and the idle hides in one sync kernel
L=4096 prefill, busy (Σ kernel dur) vs wall-span, per rank:
```
 rank    busy     span   idle    note
   0     98.6    196.2   97.6    idle shows as inter-kernel GAPS (50% busy)
   1    191.5    195.5    4.0    idle ABSORBED into _reduce_insert_kernel (96.3 ms)
   2    191.8    ~195     ~4     _reduce_insert = 97.0 ms
   3    191.6    ~195     ~4     _reduce_insert = 96.2 ms
```
`_reduce_insert_kernel` is **2.7 ms on rank0 but ~96 ms on ranks 1–3** — a cross-rank
sync point where three ranks spin-wait. Both views agree: **all ranks span ~196 ms
wall, of which only ~99 ms is real compute; ~97 ms (~50%) is idle** (launch gaps +
TP sync bubble). This is the textbook C=1 signature — latency/launch/sync-bound, not
compute-bound — and it's *why* the two big wins in the prefill daylog worked: FULL
CUDA-graphs (T4) remove the launch-gap half, and batching (T3/EC3) fills the bubble.
(Root cause of the rank0-vs-rest imbalance at the reduce/insert step — root-heavy
reduction vs rank0 doing extra serial routing work — needs deeper tracing.)

## Finding 2 — the compute map (rank0, L=4096, the un-absorbed rank)
Σ 98.6 ms GPU-kernel self time, 2068 launches:
```
component                              ms      %   launches
MoE (FP4 experts + route)           36.3   36.8%      446
attention (FA4 + sconv/rel/sink)    34.7   35.2%      988
dense GEMM (qkv/o/shared/router)    24.8   25.1%      335
glue / quant / norm                  2.7    2.8%      298
allreduce/comm (in-trace)            0.02   0.0%        1
```
Top kernels (name → identity):
```
 15.8ms  bmm_E2m1_E2m1E2m1_Fp32_... (×63)   MoE expert GEMM, FP4 (E2M1) tensor-core
 14.5ms  nvjet_sm100_tst_128x256... (×196)  dense GEMM (cuBLASLt sm100): qkv/o/shared
 13.3ms  bmm_Bfloat16_E2m1E2m1_...  (×63)   MoE expert GEMM, bf16×FP4
  9.1ms  _sconv_publish_kernel      (×132)  Inkling attention: short-conv publish
  7.9ms  _publish_input_kernel      (×132)  attention data movement
  6.2ms  nvjet_sm100_tst_...        (×57)   dense GEMM
  4.7ms  moe::dev::finalize::...    (×63)   MoE combine expert outputs
  4.6ms  _gather_norm_kernel        (×132)  attention gather + norm
  3.0ms  ...fa4flash_fwd...         (×55)   FlashAttention-4 forward (the actual attn)
  2.7ms  _reduce_insert_kernel      (×132)  KV reduce/insert (the C=1 sync point)
  2.2ms  ...fa4shearing...          (×55)   FA4 shearing (rel-pos variant)
  1.2ms  _inkling_gate_select       (×64)   MoE routing (top-6 of 256)
  1.1ms  _rel_proj_throughput       (×66)   relative-position projection (d_rel=16)
  1.0ms  _kv_kernel / _sink_epilogue(×64)   KV proj / attention sink
  0.6ms  cvt_fp16_to_fp4_sf         (×63)   activation → FP4 quant for MoE
```

## Reading the architecture off the kernels
- **MoE experts run on native FP4 tensor cores** — the `bmm_*E2m1*` kernels (E2M1 = FP4).
  Confirms the `FLASHINFER_TRTLLM` backend uses the Blackwell FP4 path, no bf16
  upconvert. ~37% of compute for top-6-of-256 routing + 2 shared experts.
- **Attention ≠ FlashAttention.** The FA4 math (`fa4flash`+`fa4shearing`) is only ~7 ms;
  the other ~24 ms of the attention bucket is Inkling's own machinery — `_sconv_*`
  (short convolution over the sequence), `_rel_proj` (relative position, `d_rel=16`,
  `rel_extent=1024` from config), `_sink_epilogue` (attention sink; `shared_expert_sink`
  in config), plus `_publish`/`_gather`/`_reduce_insert` data movement. All at **132
  launches = 2× per layer** — many tiny, movement/launch-bound kernels. This is the
  distinctive part of the model and the biggest bag of small kernels.
- **Dense GEMMs are cuBLASLt `nvjet` sm100** (qkv / o_proj / shared-expert / router),
  ~25%.
- **norm ~0, in-trace allreduce ~0**: RMSNorm is fused (`fuse_norm_quant`); TP
  all-reduce cost is hidden inside the `_reduce_insert` sync rather than a named NCCL
  kernel at this size.

## Caveats — what is and isn't trustworthy here
- **Relative shares, kernel identities, and the idle/sync structure are robust** (cross-
  validated across all 4 ranks).
- **Absolute compute magnitude and length-scaling are NOT.** A second capture at L=2048
  gave 15.1 ms rank0 compute vs 98.6 ms at 4096 — 6.5× for 2× tokens, uniform across
  buckets — which is non-physical. Cause: the async engine loop + torch profiler's
  default `active_iterations=5` schedule capture a variable amount of work around a
  single `/start_profile`→`/stop_profile` prefill. Treat the ms values as within-trace
  proportions, not calibrated per-token costs.

## Open (next)
- Pin the capture window: `--profiler-config.max_iterations=1` (+ `active_iterations=1`,
  `ignore_frontend=true`) or drive via `nsys` for a single clean forward, then redo the
  L-sweep to get real per-token scaling per bucket.
- Resolve the `_reduce_insert` rank0-vs-rest imbalance (why one rank runs ~93 ms longer
  before the sync) — is it root-heavy reduction, EP routing serialization, or a
  genuine load imbalance across the expert-parallel groups?
- Profile a decode step (out of scope for this pass) if the decode path ever matters.

```
Box: 4× B200 183 GiB, vLLM 0.1.dev18898+g93d5b2187 (inkling), NVFP4, TP=4 + EP,
--enforce-eager, torch profiler. Scripts: prof_prefill.py, parse_trace.py.
```
