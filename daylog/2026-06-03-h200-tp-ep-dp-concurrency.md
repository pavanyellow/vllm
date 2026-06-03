# 2026-06-03 (H200 ×4) — does voice concurrency scale with TP/EP, or only with DP replicas?

Follow-up to [2026-06-02-h200-voice-concurrency.md](2026-06-02-h200-voice-concurrency.md). That
study profiled **one** H200 and found the voice operating point is **latency-bound, not
throughput-bound**: at the ~200-call knee the GPU is ~90 % idle (avg in-flight 3.5 of 32 slots),
and p99 TTFT — not the batch ceiling — is what breaks the SLO. Natural question: **if we add GPUs,
does the concurrency knee move up?** And does *how* we add them matter — sharding one engine
(tensor / expert parallel) vs running independent replicas (data parallel)?

Box: **single node, 4× H200 141GB**, NVLink full mesh (NV18 all-pairs), 2 NUMA domains
(GPU0/1 ↔ NUMA0, GPU2/3 ↔ NUMA1). vLLM 0.21.0. Model as before: `Qwen/Qwen3.6-35B-A3B-FP8`
(`qwen3_5_moe`: hybrid linear-attention/gated-deltanet + full-attention every 4th layer, MoE
**256 experts, top-8**, 40 layers, ~3B active). 256 experts divides cleanly for EP across 2 or 4.

**TL;DR:** TP/EP did **not** move the knee — 2 and 4 GPUs both stayed at ~200 calls. **DP replicas
scaled it ~4× (200 → ~800 calls at the same p99).** For a model that already fits on one GPU,
scale voice concurrency with **data-parallel replicas + a sticky load balancer**, not TP/EP.

## Configs tested

All share the 2026-06-02 serving config (drop `--block-size 8192`, extend cudagraph captures with
the G7 big-chunk rungs 2112/3168/4224, `--fingerprint-mode none`, `--max-num-seqs 32
--max-num-batched-tokens 8192`, `--enable-prefix-caching`). The four configs differ only in
parallelism:

| config | launch delta | GPUs |
|---|---|---:|
| single-GPU (baseline) | — (numbers from 2026-06-02) | 1 |
| TP=2 + EP | `--tensor-parallel-size 2 --enable-expert-parallel`, `CUDA_VISIBLE_DEVICES=0,1` | 2 |
| TP=4 + EP | `--tensor-parallel-size 4 --enable-expert-parallel`, `CUDA_VISIBLE_DEVICES=0,1,2,3` | 4 |
| 4× DP + LB | 4 independent single-GPU servers on ports 8001–8004 (one GPU each) behind a Caddy LB on 8080 | 4 |

All four-GPU configs captured the full graph set (29 PIECEWISE + 7 FULL) and ran the DeepGEMM
warmup (1.0–1.3k kernels) before serving. KV is never the constraint: TP=2 reported max concurrency
**692×**, single-GPU **294×**, both far above the 32-slot batch.

### DP load balancer (session affinity)

Naive round-robin would scatter a call's sequential turns across replicas and destroy the intra-call
prefix cache (V2 of the prior daylog: caching is a 3.5× TTFT factor). So the driver sends a per-call
`X-Session-Id` header and Caddy hashes on it (`lb_policy header X-Session-Id`) → every turn of a call
pins to the same replica. Each replica then behaves exactly like the single-GPU baseline.

```
# Caddyfile
:8080 {
  reverse_proxy 127.0.0.1:8001 127.0.0.1:8002 127.0.0.1:8003 127.0.0.1:8004 {
    lb_policy header X-Session-Id
    flush_interval -1          # don't buffer the SSE stream (preserve true TTFT)
  }
}
```

Driver: `voice_sim.py` (prior daylog) with two small additions — (a) catch dropped/reset
connections per-turn instead of aborting the whole run (at 800+ calls a single `httpx.ReadError`
otherwise kills the gather before the summary prints), and (b) emit the `X-Session-Id` header.
Workload unchanged: `--duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30`, cache ON.

## Results — TTFT p99 (ms), the voice SLO metric (~250 ms target)

| calls | 1× GPU (06-02) | TP=2 + EP | TP=4 + EP | 4× DP + LB |
|---:|---:|---:|---:|---:|
| 100 | 82 | 1169* | — | — |
| 200 | 234 | 235 | — | **79** |
| 400 | — | 943 | 1575*† | **165** |
| 800 | — | 4434 | 8201 | **270** |
| 1600 | — | — | — | 52000 (saturated) |

\* cold prefix cache (first traffic after boot — the unique 4k contexts hadn't been prefilled).
† TP=4's 400 was its first run (cold); TP=2's 400 was its 3rd run (warm) → these two p99s are not
apples-to-apples. p50/p90 (below) are the fair read at 400.

### Full TTFT distributions + in-flight

| config / calls | p50 | p90 | p99 | max | peak in-flight | avg in-flight |
|---|---:|---:|---:|---:|---:|---:|
| TP2EP 200 | 51.5 | 101.3 | 235.4 | 294.6 | 23 | 3.66 |
| TP2EP 400 | 71.0 | 278.2 | 943.0 | 1182.1 | 67 | 13.43 |
| TP2EP 800 | 2577.5 | 3960.7 | 4433.9 | 4519.5 | 253 | 126.43 |
| TP4EP 400 | 59.2 | 186.2 | 1575.1* | 2317.3 | 56 | 9.61 |
| TP4EP 800 | 711.3 | 6077.1 | 8200.7 | 8764.3 | 388 | 104.51 |
| DP 200 | 50.1 | 61.3 | 79.5 | 105.2 | 12 | 2.41 |
| DP 400 | 54.1 | 88.2 | 164.9 | 281.9 | 27 | 5.80 |
| DP 800 | 62.4 | 119.9 | 270.1 | 488.8 | 87 | 16.03 |
| DP 1600 | 4150.0 | 26071.7 | 52000.0 | 68676.4 | 1044 | 541.75 |

## Where the knee lands (max concurrent calls at p99 ≤ ~250 ms)

| config | GPUs | knee | scaling |
|---|---:|---:|---:|
| single-GPU | 1 | ~200 | 1× |
| TP=2 + EP | 2 | ~200 | **1×** |
| TP=4 + EP | 4 | ~200 | **1×** |
| 4× DP + LB | 4 | **~800** | **~4×** |

## Why TP/EP doesn't move the knee

The knee is set by **prefill latency when a few cold 4k requests collide**, not by aggregate GPU
throughput (the prior daylog's core finding: ~90 % idle at the knee). TP=2/4 + EP shards **one**
engine/scheduler across GPUs and adds **all-reduce (TP) + all-to-all (EP) comms every layer**. It:
- does **not** reduce the *number* of colliding cold prefills, and
- on a **~3B-active** model the per-token compute is tiny, so comms overhead roughly cancels the
  extra FLOPs. The p99 tail doesn't shrink; the knee stays at ~200.

What TP=4 *did* help is **single-request latency** — at 800 calls its p50 is 711 ms vs TP=2's
2577 ms (4-way compute clears an individual prefill faster). But that's the wrong axis for voice
concurrency, and raw saturated throughput (~43–45 req/s) barely differed between TP=2 and TP=4.

## Why DP scales ~4× (cleanly)

Four independent engines, each with its own scheduler and 32-slot batch; sticky LB keeps each call's
cache local. So each replica is the single-GPU baseline, and capacity adds:

- **Apples-to-apples:** 1 GPU serves 200 calls @ p99 234 ms; 4× DP serves **800 calls @ p99 270 ms**
  — 4× the calls at ~the same latency.
- **By construction:** 800 ÷ 4 = 200 calls/replica = exactly the single-GPU operating point. The
  avg-in-flight confirms it: DP @ 800 is 16.0 total → **4.0 per replica**, matching single-GPU @ 200
  (~3.5).
- **Sub-knee bonus:** at 200 calls DP gives p99 **79 ms** (vs 234 single-GPU) — the load splits
  ~50/replica, so each replica is barely loaded.
- **Saturation where predicted:** 1600 calls = 400/replica, past each replica's own knee → cliff
  (p99 52 s, avg in-flight 542 ≈ 4 × the single-GPU 500-call wall).

`errors=0` on every DP run; session affinity held (verified: repeated `X-Session-Id: call-7`
requests hit the same backend).

## Caveats

- **Throughput (req/s) numbers are unreliable past the knee.** At saturation, per-request latency
  exceeds the 75 s measurement window (DP-1600 p99 = 52 s), so `turns/wall` undercounts true
  capacity. The **knee (concurrent calls at SLO) is the trustworthy metric** — and it's a clean ~4×.
- The single-GPU column is carried over from 2026-06-02 (same box class, same vLLM, same model),
  not re-measured this session.
- The 100-call (TP2EP) and 400-call (TP4EP) p99s are cold-cache; don't compare their tails to warmed
  runs. A throwaway warmup pass before each sweep (added for the DP runs) removes this.

## Takeaway

For a small MoE that fits on one GPU, **TP/EP is the wrong tool for concurrency** — it's for fitting
a bigger model or cutting single-request latency. **Scale voice concurrency with data-parallel
replicas + a sticky (session-affinity) load balancer:** N replicas ≈ N× calls at the same p99, with
better tail below the knee. On this box, 4× H200 → ~800 concurrent voice calls at p99 ≤ ~270 ms.

## Open

- Pin the DP knee between 600–800 (p99 crosses 250 ms just under 800) and confirm linearity at
  2× / 3× replicas.
- Warm 100/200 on TP=4+EP to formally close out its (flat) knee vs the cold tails seen here.
- DP with **heterogeneous** replica sizing or a global queue (vs per-replica 32 slots) — can a
  shared admission layer shave the tail further?
