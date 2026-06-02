# 2026-06-02 (H200) — voice-traffic concurrency & TTFT profiling

First study on **H200 141GB** (vs the H100 80GB of the prior daylogs). Two parts:
1. validated the **G7 big-chunk-graph** prediction that was OOM-blocked on H100, and
2. profiled **realistic voice traffic** (N independent calls, sparse bursty turns)
   to find the real per-GPU call capacity and the cost/economics — using a new
   driver, `voice_sim.py` (knobs documented below).

Box: H200 141GB HBM3e (~4.8 TB/s), vLLM 0.21.0, single GPU. Model as before:
`Qwen/Qwen3.6-35B-A3B-FP8` (hybrid gated-deltanet + attention MoE, ~3B active).

## Serving config

Note this differs from the H100 README default: on H200 we **drop `--block-size 8192`**
(the hybrid mamba/attention page coupling pads the mamba page ~6.8× and OOMs even on
141GB during the profiling forward — same failure mode as H100, just at a higher
ceiling). Instead we keep the stock block (1056) and **extend the cudagraph capture
list to the block-multiples 2112/3168/4224** — the G7 rungs. Plus `--fingerprint-mode
none` for the customer-facing endpoint (nulls the `system_fingerprint` that otherwise
leaks vLLM version + a config hash).

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen/Qwen3.6-35B-A3B \
  --port 8080 --fingerprint-mode none \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}' \
  --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,128,256,384,512,640,768,896,1024,1152,1200,2112,3168,4224]}'
```

## Server stats / observability

vLLM exposes a Prometheus `/metrics` endpoint (unauthenticated; open here — would be
firewalled in a locked-down customer deployment). Two scrapes used throughout:

**Cumulative tokens processed** (counters; reset per server instance):
```bash
curl -s http://127.0.0.1:8080/metrics | grep -E "^vllm:(prompt|generation)_tokens_total"
# vllm:prompt_tokens_total{...}     1.6316669e+07
# vllm:generation_tokens_total{...} 150806.0
```

**Live load snapshot** (running vs queued, KV pressure, cache hit rate):
```bash
curl -s http://127.0.0.1:8080/metrics | grep -E \
  "^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|prefix_cache_(hits|queries)_total)"
```

`num_requests_waiting` is the queue depth — the saturation tell: ~0 below the wall,
climbing once offered load exceeds the ~120k tok/s prefill ceiling (V5). The engine
also logs a periodic line with the same fields:
```bash
grep "Avg prompt throughput" vllm-8080.log | tail -1
# Avg prompt throughput: X tok/s, Avg generation throughput: Y tok/s,
# Running: R reqs, Waiting: W reqs, GPU KV cache usage: Z%, Prefix cache hit rate: P%
```

## V0 — G7 validated on H200: big-chunk graphs collapse the cold 3k–5k plateau

`ttft_sweep.py` (clean, `cache_frac=0`), captures ≤1200 vs captures + {2112,3168,4224}:

| in_len | p50 graphs≤1200 | p50 +2112/3168/4224 | Δ |
|---:|---:|---:|---:|
| 2200 | 107.3 | 57.9 | −46% |
| 3100 | 116.8 | 72.5 | −38% |
| 4000 | 116.2 | 84.0 | −28% |
| 5000 | 116.5 | 96.5 | −17% |

The ~107–117 ms cold plateau (the un-graphable big chunk of the 2-pass) is gone; the
band rises smoothly 58→96 ms. The 2112 rung is the −46% cliff at L=2200. All 29
PIECEWISE captures (incl. the 3 big rungs) fit on H200, no OOM. Cached re-serve
(`--cache-frac 0.9`) is ~22–55 ms across 0–5k — confirms the G6 sub-50 ms headline.

## Voice-traffic model (`voice_sim.py`)

The synchronized-burst sweep (`ttft_sweep --concurrency C`) is the **pessimistic
worst case** — all calls hitting their LLM turn at the same instant. Real voice
isn't that: a single call is sequential (never overlaps itself), and between turns
there's ~8–16 s of human-speak + TTS playback + silence. The LLM is "busy" only
~0.2 s per turn, so each call occupies a slot ~1–3 % of the time. The metric that
matters is **simultaneous in-flight requests**, not active-call count.

`voice_sim.py` simulates N independent calls: staggered starts, per-call sequential
turns with random 8–16 s gaps, per-call conversation prefix reused each turn
(intra-call cache hit). It reports the TTFT distribution **and** the real peak /
time-averaged in-flight concurrency, plus actual `prompt_tokens` (from usage).

### Knobs
- `--calls N` — number of independent active calls.
- `--duration S` — wall-clock seconds (calls loop turns until then; ~6 turns/call at default gaps).
- `--gap-min / --gap-max` — inter-turn gap seconds (default 8–16) = the duty cycle.
- `--ctx-min / --ctx-max` — per-call reference-context target tokens (0 = short chat; e.g. 3000 5000 for RAG-style 3–5k input). Length verified via `prompt_tokens`.
- `--groups "30,30"` — % of calls sharing each system prompt (identical block → cross-call cache hit); remainder unique per call. `"30,30"` = two 30 % shared tenants + 40 % unique.
- `--no-cache` — prepend a fresh nonce each turn to bust the prefix cache (full context re-prefill every turn).

```bash
python voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30
```

## V1 — realistic traffic never approaches the batch ceiling

Short chat turns (~400-tok context), 75 s:

| | 50 calls | 100 calls |
|---|---:|---:|
| TTFT p50 / p99 (ms) | 29.2 / 63.9 | 26.0 / 44.1 |
| peak in-flight | 5 | 7 |
| avg in-flight | 0.47 | 0.95 |

100 active calls → **peak 7** simultaneous in-flight (avg ~0.95). Effective
concurrency ≈ 1 % of call count. The 32-slot batch is idle the vast majority of
the time; concurrency scales ~0.0095 in-flight per call.

## V2 — caching is irrelevant at short ctx, decisive at 3–5k

Short ctx: cache on/off is within noise (both ~25–29 ms p50) — a few-hundred-token
context re-prefills in ~25 ms on the captured graphs, nothing for the cache to save.
At **3–5k ctx** (p50 ~4.1k input tok), 100 calls:

| | cache ON | cache OFF |
|---|---:|---:|
| TTFT p50 | 44.7 | 157.2 |
| avg in-flight | 1.51 | 3.42 |

Cache cuts median TTFT ~3.5× **and** GPU occupancy ~2.3× (Little's Law: no-cache
requests are "busy" longer → more pile up → ~2× the call capacity). Caching earns
its keep only once contexts are in the thousands of tokens.

## V3 — the cold first-turn tail scales with *distinct* contexts, not calls

With per-call-unique 4k contexts, every call's first turn is a cold 4k prefill
(~1-in-6 turns) — that's what fills p90/p99. Concentrating traffic onto shared
system prompts (`--groups 30,30`: two 30 % tenants share a cached block, 40 % unique)
primes each shared block **once globally**, so later calls in a group hit on their
first turn too. 100 calls, 4k ctx, cache ON:

| | all-unique (per-call) | mix 30/30/40 |
|---|---:|---:|
| input tokens p50 | 4114 | 3984 |
| TTFT p50 | 44.7 | 42.4 |
| TTFT p90 | 135.1 | 56.6 |
| TTFT p99 | 281.7 | 82.0 |
| TTFT max | 306.0 | 90.0 |
| avg in-flight | 1.51 | 1.20 |
| peak in-flight | 14 | 9 |

Cold prefills drop from ~100 (one per call) to ~42 (40 unique + 2 group primes):
**p99 −71 %, p90 −58 %.** p50 unchanged — the tail is fixed, not the median.

## V4 — the knee, and what's actually binding

Mix 30/30/40, 4k ctx, scaling calls:

| | 100 calls | 200 calls |
|---|---:|---:|
| throughput | 6.8 req/s | 13.1 req/s |
| TTFT p50 | 42.4 | 48.9 |
| TTFT p99 | 82.0 | 233.6 |
| peak in-flight | 9 | 20 |
| avg in-flight | 1.20 | 3.49 |

At the ~200-call knee, **avg in-flight is 3.5 of 32 (~11 % GPU utilization)** — so
the system is **latency-bound, not throughput-bound**. p99 climbs to ~234 ms (cold
unique-context prefills colliding) *before* the slot count saturates. For a voice
SLO of p99 ≤ ~250 ms, that's ~**200 concurrent active calls per H200**; raw
throughput could go several-fold higher with a looser SLO or backfill traffic.

## V5 — past the knee: the saturation cliff, and why bigger queues don't help

Pushing the mix to 500 calls (`--max-num-seqs 32 --max-num-batched-tokens 8192`):

| | 100 | 200 | 500 |
|---|---:|---:|---:|
| throughput | 6.8 req/s | 13.1 | 29.1 |
| TTFT p50 | 42.4 | 48.9 | 349.8 |
| TTFT p99 | 82.0 | 233.6 | 6,188 |
| peak in-flight | 9 | 20 | 206 |
| avg in-flight | 1.20 | 3.49 | 52.96 |

500 calls offer ~170k tok/s vs the GPU's ~120k tok/s prefill ceiling → unbounded
queue, p99 = **6.2 s**. The avg-in-flight jump (3.5→53) is **Little's Law**
(`in-flight = throughput × latency`): throughput ×2.2 (13→29 req/s) and per-request
latency ×6.9 (~0.26→~1.8 s, mostly queue wait) multiply to ×15. Peak 206 =
~32 running + ~174 queued. Below capacity requests pass through (in-flight ∝ calls);
above it they stack up (in-flight = queue depth).

**Do more slots / a bigger batch budget move the wall?** Re-ran with both doubled
(`--max-num-seqs 64 --max-num-batched-tokens 16384`):

| | 200 (32/8192) | 200 (64/16384) | 500 (32/8192) | 500 (64/16384) |
|---|---:|---:|---:|---:|
| throughput | 13.1 req/s | 13.1 | 29.1 | 29.6 |
| TTFT p50 | 48.9 | 50.6 | 349.8 | 159.4 |
| TTFT p90 | 107.3 | 106.5 | 3,875 | 2,159 |
| TTFT p99 | 233.6 | **686.4** | 6,188 | 3,479 |
| TTFT max | 283.1 | 1,167.9 | 6,461 | 3,724 |

- **Throughput ceiling unchanged** (29.1→29.6 req/s, ~120k tok/s). Prefill is
  **compute/bandwidth-bound on the 32 GB MoE weight read**, NOT gated by
  `max-num-batched-tokens`. The ~350-call wall doesn't move.
- **At 500** (over capacity) latency ~halved (p99 6.2→3.5 s) — bigger steps drain the
  backlog more efficiently — but still multi-second, still unusable.
- **At 200** (the SLO knee) p99 got **worse** (234→686 ms): the 16k budget packs ~4
  concurrent 4k prefills into one step, so a request sharing that megabatch waits
  longer for its first token → fatter tail. **Net negative at the voice operating point.**

**Takeaway:** for the voice SLO keep `--max-num-seqs 32 --max-num-batched-tokens 8192`
(tighter p99 at the knee). Doubling only helps over-capacity batch-style traffic, and
never raises the throughput ceiling. Capacity grows only via **smaller context, more
prompt sharing, or more GPUs** — not bigger queues.

## Capacity & economics (per H200, $5/hr, ~6-turn ~1-min calls, 4k ctx)

Steady-state at the knee:
- **~10–12k calls/hr**, ~60k turns/hr, **~245 M input tok/hr** (~68k tok/s); output ~1.9 M/hr (negligible, ~32 tok/turn).
- Workload is **108:1 input:output** — overwhelmingly prefill-bound.
- Compute cost ≈ **$0.026 / 1M input tok**. At $2–3/M billing that's ~1 % of revenue (~77–115× markup, ~99 % gross margin).
- Break-even ≈ **~1,640 calls/day** @ $3/M (~0.7 % of one GPU's ~240k-call/day capacity; the GPU pays its $120/day in ~13 min at the knee).
- Scales linearly: 10k concurrent calls ≈ 50 H200 ≈ $250/hr.

Biggest levers: **context size** (cost & revenue both scale with input tokens, cost ~40× slower), **shared-prompt fraction** (fewer cold prefills → flatter p99 → more calls/GPU), and **filling the ~89 % idle GPU** with batch traffic.

## Files
- `voice_sim.py` — voice-traffic concurrency/TTFT driver (knobs above; chat-completions, async, reports in-flight concurrency + `prompt_tokens`).
- `ttft_sweep.py` — TTFT vs input-length sweep (used for V0).

## Open
- Sweep call count 250/300/400 to pin the exact p99-SLO crossing and the peak→32 saturation point.
- Re-baseline with real call-length / gap distribution (current ~1-min, 6-turn calls are short vs typical voice).
- Map tail vs unique-fraction (10/10/80 … 45/45/10).
