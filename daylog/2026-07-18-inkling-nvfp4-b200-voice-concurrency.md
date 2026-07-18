# 2026-07-18 (4× B200) — Inkling-NVFP4 voice-traffic concurrency & the prefix-cache/KV wall

Ran the qwen voice-concurrency study (`voice_sim.py`, staggered independent calls, 8–16 s
inter-turn gaps, per-call context reuse) against **`thinkingmachines/Inkling-NVFP4` on
4× B200**, plus a deep dive on *why* the node saturates. Companion to the prefill/TTFT
(`…-b200x4.md`) and kernel-map (`…-b200-kernels.md`) daylogs.

**Headline:** Inkling saturates voice traffic **~4–16× earlier per GPU than qwen-35B**,
and the entire story is **KV capacity**. Because it's a ~1T-param / 37B-active model, the
GPU holds only **~38.8k KV tokens (~10 concurrent 3–5k contexts)**. When calls have
*distinct* contexts, they evict each other between turns → **0% cache hit → every turn
re-prefills the full 4k → ~20k tok/s of recompute** that pins the node. Route all calls
through a *shared* context and prefill collapses **460×** (20k → 43 tok/s) and it stays
flat. The residual latency past that is the model's intrinsic ~50 ms/forward cost, not
caching.

## Setup
- Model served with thinking **OFF** — Inkling is a reasoning model; per-request
  `chat_template_kwargs={"reasoning_effort":"none"}` (maps via the chat template's
  `{none:0.0 … high:0.9(default) … max:0.99}`). Verified: reasoning empty, direct short
  spoken reply, `finish=stop` in ~38 tok (default fills 64 tok with hidden reasoning and
  never answers — would wreck the sim). `voice_sim.py` patched to send it.
- `--max-model-len 8192`, **KV = 38,868 tokens** (4.74× @ 8192). Two server variants:
  **U** = default `max_num_seqs` (uncapped, ~1024); **S32** = `--max-num-seqs 32
  --max-num-batched-tokens 8192`. Prefix caching on. `/v1/chat/completions`, `ctx 3–5k`,
  `max_tokens 64`, 60–75 s wall. Prefill measured from `/metrics` `prefix_cache_queries
  − hits` deltas.

## V1 — realistic voice traffic, groups 30/30 (server U)
`--ctx-min 3000 --ctx-max 5000 --groups 30,30` (two 30% shared tenants + 40% unique),
75 s:

| calls | req/s | input p50 | TTFT p50 | p90 | p99 | max | peak inflight | avg inflight |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 50  | 3.4  | 4311 | 34.3  | 49.4 | 260 | 319 | 7  | 0.87 |
| 100 | 6.4  | 3849 | 40.2  | 207  | 346 | 378 | 19 | 2.58 |
| 200 | 12.1 | 4044 | 287.3 | 464  | 779 | 959 | 70 | 17.77 |

vs the qwen-35B B200 daylog (single GPU): 200 calls was p50 49 ms / p99 234 / avg
inflight 3.5. Inkling at 200 is p50 287 / avg inflight **17.8 (~5×)**.

### Why avg-inflight jumps ~7× for a 2× call bump (it's real, not a bug)
`avg inflight = busy_s/wall`, `req/s = turns/wall` ⇒ `avg inflight ÷ req/s = mean
per-request latency`. 100 → 0.40 s, 200 → **1.47 s** (×3.6). In-flight = throughput ×
latency, so ×1.9 × ×3.6 = ×6.9. We crossed the saturation knee between 100 and 200.
**Server logs prove the cause:** during the 200-call run `Waiting: 0` throughout (no
queue) and `GPU KV cache usage` peaked ~26% (not KV-blocked) while prompt throughput hit
~24–28k tok/s with `Running` 30–40. So it's **in-batch compute dilution** — the
chunked-prefill scheduler admits everything as *Running* and time-slices ~24–28k tok/s
of prefill across them, so each request's own prefill stretches. Aggregate prefill tops
out ~**24–28k tok/s** (vs qwen's ~120k on one GPU) — the ~1T model's ~96 ms/4k-prefill
(nsys) is the ceiling.

## V2 — the real driver is DISTINCT contexts, not calls (server S32, 100 calls)
Held calls=100, ctx 3–5k; varied only how many *distinct* contexts exist. Measured real
computed prefill (`queries − hits`):

| sharing | distinct ctx | **hit rate** | **computed prefill** | TTFT p50 / p99 | avg inflight |
|---|--:|--:|--:|--:|--:|
| all-unique  | 100 | **0.0%**  | **19,994 tok/s** | 343 / 822 ms | 9.46 |
| groups 30,30 | ~42 | 88.8%    | 2,747 tok/s      | 40 / 312 ms  | 2.17 |
| all-shared   | 1   | **99.8%** | **43 tok/s**     | 34.5 / 45 ms | 1.69 |

**This is the whole story.** All-unique gets **0% hit** — not because caching is off, but
because 100 distinct 4k contexts can't fit in 38.8k KV (~10 contexts), so each call's
blocks are **evicted during its 8–16 s gap** and turn-N re-prefills the full 4k. That
20k tok/s recompute *is* the saturation. A genuinely shared block stays hot on its own
(LRU keeps it — 99.8%), so prefill drops **460×** and p99 flattens to 45 ms.

## V3 — shared context + ~256 fresh tokens/turn (server S32)
The realistic "shared system prompt, one turn at a time" case: one shared 4k context +
a fresh ~256-tok turn each request (`--groups 100 --turn-tokens 256`), 60 s:

| calls | req/s | hit rate | computed prefill | TTFT p50 | p90 | p99 | peak inflight | avg inflight |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 50  | 3.1 | 93.3% | 847 tok/s   | 192  | 316  | 344  | 13  | 2.53 |
| 100 | 5.7 | 93.3% | 1,543 tok/s | 253  | 389  | 477  | 26  | 8.81 |
| 200 | 8.0 | 93.3% | 2,185 tok/s | **4511** | 6538 | 7080 | 104 | 56.01 |

Prefill stays **~15–20× below all-unique** (hit 93.3% = 256/4081 fresh). 50–100 calls
track EC5 exactly (~250 ms at C≈8). **But 200 calls still cliffs to p50 4.5 s** — here
`max_num_seqs=32` caps concurrency so the excess *queues* (client inflight 56 > 32
running). Even with near-free prefill, the ~1T model's raw forward-pass throughput can't
clear 200 calls' worth of turns → queue wait dominates. **Voice capacity is model-size-
bound, not cache-bound, once you're sharing.**

## Capacity read (per 4× B200 node, thinking off, 3–5k ctx)
- For a voice SLO of **p99 ≤ ~250 ms**: ~**50 calls/node** with mixed contexts (V1),
  rising sharply with prompt sharing. qwen-35B did ~200 on a *single* GPU → Inkling is
  ~**16× fewer calls/GPU**, consistent with being ~10–30× larger.
- The binding resource is **KV capacity → cache retention**, then raw forward throughput.

## Keeping contexts hot / not evicted — the levers
Eviction is strict **LRU** in this build; there is **no pin/priority API** (`cache_salt`
only *isolates* tenants). Options, best first:
1. **`--kv-offloading-size <GiB>` (+ `--kv-offloading-backend native|lmcache`).** Spill
   evicted prefix KV to CPU RAM (~1.7 TB here) instead of dropping it. A GPU-KV miss then
   becomes a CPU→GPU copy (~10–15 ms for a 4k ctx) instead of a ~96 ms recompute —
   eviction stops meaning "lose it." Directly attacks the all-unique 20k-tok/s wall.
   *(A/B pending.)*
2. **More GPU KV** — `--kv-cache-memory` higher / fewer graphs / more GPUs. Root cause is
   the ~1T model leaving only 38.8k tokens.
3. **Route through shared prefixes** — genuinely shared blocks stay hot via LRU for free
   (V2 all-shared). No pin needed.

## V4 — KV offloading A/B: buffer size is everything (128 GiB fails, 512 GiB works)
Relaunched with `--kv-offloading-size {128,512} --kv-offloading-backend native` (+ the
tuned 6144 graph ladder), same S32 scheduler. vLLM logs `CPUOffloadingSpec`; transfer
counters via `/metrics` `vllm:kv_offload_total_bytes_total{transfer_type}`.

**128 GiB: write-only, useless.** All-unique 100 calls: hit rate still **0.0%**,
recompute 22,341 tok/s, TTFT p50 284 ms. Counters: **GPU→CPU 1.3 TB, CPU→GPU 0 bytes**
— 1.3 TB spilled into a 128 GiB buffer = ~10× wrap; the CPU tier itself LRU-evicts a
call's blocks during its 8–16 s gap, so nothing is ever read back.

**512 GiB: read-back turns on and the wall moves.** `CPU_to_GPU` went 0 → **33.8 GB**:

| arm (100 calls, ctx 3–5k) | hit rate | computed prefill | TTFT p50/p99 | avg inflight |
|---|--:|--:|--:|--:|
| all-unique, no offload (V2)     | 0.0%  | 19,994 tok/s | 343 / 822 ms | 9.46 |
| all-unique, offload **128 GiB** | 0.0%  | 22,341 tok/s | 284 / 672 ms | 8.48 |
| all-unique, offload **512 GiB** | **60.5%** | **9,570 tok/s** | **44 / 434 ms** | **2.83** |
| groups 30,30, no offload (V2)   | 88.8% | 2,747 tok/s  | 40 / 312 ms  | 2.17 |
| groups 30,30, offload **512 GiB** | **99.2%** | **206 tok/s** | **37 / 60 ms** | **1.61** |
| shared+256-fresh, offload 512   | 93.3% | 1,747 tok/s  | 56 / 143 ms  | 2.55 |

- Cleanest same-ladder pair (all-unique, 128 vs 512 GiB): **TTFT p50 284 → 44 ms
  (6.5×)**, recompute halved, in-flight 3× lower.
- **Groups 30,30 (the production-like mix): p99 312 → 60 ms and recompute 2,747 → 206
  tok/s (13×)** — the 40% unique callers now reload from CPU instead of recomputing.
  For this workload, a big-enough CPU spill buffer effectively *solves* the KV wall.
- Shared+256-fresh: unchanged caching (93.3% both ways — the fresh 256 are genuinely
  new; CPU→GPU stayed flat during this arm since the shared block never leaves GPU).
- Caveat: V2 baselines ran on the default ≤512 graph ladder, offload arms on the 6144
  ladder — hit-rate/recompute comparisons are ladder-independent; TTFT deltas vs V2
  are partly graphs. The 128-vs-512 pair is confound-free.
- Sizing rule: buffer must hold **all live distinct contexts** for at least one
  inter-turn gap: ≈ contexts × ctx_tokens × KV-bytes/token, with ~10× headroom vs the
  naive estimate (observed churn). Host RAM here is 1.7 TB — 512 GiB is cheap.

## V5 — production-faithful outbound sim (`voice_sim_prod.py`)
The earlier sim was too easy in places (immortal calls = no churn, no tool calls, no
barge-in) and too harsh in others (all-unique 4k contexts, every call always active).
New driver models an **outbound dialer**: N lines looping dial(ring 5–15 s) → **10%
connect** (standard outbound) → ~6-turn conversation → hangup → redial with a **new
caller**. Context = **shared 3k tenant prompt + unique 400-tok caller record** per
connect (steady cold-prefill arrival from churn). **Tool turns p=0.5** (two LLM
round-trips with a 200-tok tool result injected), **barge-in p=0.15** (stream cancelled
after a few tokens), **history grows** turn-over-turn, thinking off. Reports TTFT and
**TTFS** (time-to-first-sentence ≈ 25 tok — what the caller actually hears).

Server: S32 + 512 GiB KV offload + 6144 graphs. 120 s per level:

| dialed | ~active convos | TTFT p50/p99 | TTFS p50/p99 | hit | computed prefill | avg/peak inflight |
|--:|--:|--:|--:|--:|--:|--:|
| 500  | ~50  | 71 / 124 ms  | 363 / 529 ms   | 95.9% | 2,683 tok/s  | 6.5 / 24 |
| 1000 | ~100 | 138 / 590 ms | 593 / 1,142 ms | 87.4% | 15,834 tok/s | 21.3 / 64 |
| 2000 | ~200 | 11,377 / 18,904 ms | 12,142 / 19,585 ms | 67.7% | 52,164 tok/s | 395 / 902 |

- **dialed=500 (~50 live conversations) is the healthy operating point** — p99 TTFT
  124 ms, TTFS p99 529 ms, prefill 2.7k tok/s, GPU mostly idle.
- **The knee is between 500 and 1000**: at 1000 the hit rate slips (record churn +
  tool-result injections outpace cache), computed prefill jumps 6× toward the ~25k
  ceiling, TTFS p99 crosses 1 s.
- **2000 is a cliff**: offered prefill 52k tok/s ≈ 2× sustainable → unbounded queue,
  11 s median TTFT, 902 in flight. Same Little's-Law blowup as V1, now with realistic
  traffic shape.
- Capacity claim for this model/node with production traffic: **~500 dialed lines
  (~50 concurrent conversations) per 4×B200 node** at a voice SLO (TTFS p99 ≤ ~600 ms);
  ~2× more with a looser SLO. Tool calls ≈ 1.4× the LLM requests per turn (llm_reqs vs
  turns), already included.

## Open / next
- Re-run V3 at 200 on server **U** (uncapped) to separate queue-wait (from `max_num_seqs
  32`) from true compute saturation.
- Scheduler sweep: `--max-num-seqs {8,16,32,64}` p99 at the knee (qwen found tighter =
  better tail).
- Fill the 500–1000 gap (750 dialed) to pin the SLO crossing; longer runs (10 min) for
  steady-state cache churn.

```
Box: 4× B200 183 GiB, vLLM 0.1.dev18898+g93d5b2187 (inkling), NVFP4, TP=4 + EP,
thinking OFF (reasoning_effort=none). Drivers: voice_sim.py (+ voice_sim256.py for
--turn-tokens), measure.sh (metrics-delta prefill).
```
