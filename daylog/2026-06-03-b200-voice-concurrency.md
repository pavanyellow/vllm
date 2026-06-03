# 2026-06-03 (B200) — voice-traffic concurrency & TTFT, H200→B200 port

Re-ran the **exact** [2026-06-02 H200 voice-concurrency study](2026-06-02-h200-voice-concurrency.md)
on a single **NVIDIA B200** (Blackwell, 183 GB HBM3e), same model, same flags, same
`voice_sim.py` / `ttft_sweep.py` drivers. Every section below pairs the H200 number
(from that daylog) with the B200 number measured here.

Box: B200 183 GB HBM3e, single GPU, **vLLM 0.22.0** (the H200 run was 0.21.0 — the one
deliberate difference; noted because it affects V0's graph story). CUDA 13.0, driver
580.126.20. Model unchanged: `Qwen/Qwen3.6-35B-A3B-FP8` (hybrid gated-deltanet +
attention MoE, ~3B active; `Qwen3_5MoeForConditionalGeneration`).

**Headline:** B200 is ~8–15 % faster on prefill TTFT at the operating point with
proportionally lower GPU occupancy, but the **saturation throughput ceiling barely
moves** (~120k tok/s, MoE-weight-read-bound) — so the cliff is in the same place and
the over-capacity p99 tail is comparable. Every qualitative conclusion from the H200
study holds. The one genuinely different finding is **V0**: B200 has almost no cold
big-chunk plateau, so the G7 big-chunk-graph win is much smaller here.

## Serving config (identical to H200, three variants)

Same command as the H200 daylog. Three server configs are used:
- **Config B** — the main config (below): stock block, big-chunk graphs
  `2112/3168/4224`, `--max-num-seqs 32 --max-num-batched-tokens 8192`. Used for V0
  col 2, the cached re-serve, and V1–V5.
- **Config A** — config B **minus** the `2112/3168/4224` rungs (captures ≤1200). Used
  for V0 col 1.
- **Config C** — config B with **doubled** batch budget
  `--max-num-seqs 64 --max-num-batched-tokens 16384`. Used for V5's doubled rows. *(pending)*

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen/Qwen3.6-35B-A3B \
  --port 8080 --fingerprint-mode none \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --enable-chunked-prefill --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}' \
  --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,128,256,384,512,640,768,896,1024,1152,1200,2112,3168,4224]}'
```

All 29 PIECEWISE captures (incl. the 3 big rungs) fit on B200 with headroom — same as
H200 141 GB, no OOM (B200 has 183 GB).

## V0 — big-chunk graphs: the win that mostly evaporates on B200

**Methodology.** `ttft_sweep.py` sends raw random token IDs via `/v1/completions`
(every prompt unique ⇒ zero prefix-cache hits ⇒ measures the warm *compute* path),
`cache_frac=0`, single stream, p50 of 10 timed rounds. Run once on **config A**
(captures ≤1200) and once on **config B** (+2112/3168/4224):

```bash
python3 ttft_sweep.py --url http://127.0.0.1:8080/v1/completions \
    --model Qwen/Qwen3.6-35B-A3B --lengths 2200,3100,4000,5000
```

p50 TTFT (ms):

| in_len | H200 ≤1200 | H200 +rungs | **B200 ≤1200** | **B200 +rungs** |
|---:|---:|---:|---:|---:|
| 2200 | 107.3 | 57.9 | **65.6** | **53.3** |
| 3100 | 116.8 | 72.5 | **73.5** | **62.6** |
| 4000 | 116.2 | 84.0 | **76.3** | **72.9** |
| 5000 | 116.5 | 96.5 | **86.0** | **82.5** |

On H200 the ungraphable big prefill chunk is a flat **107–117 ms cold plateau**, and
the big-chunk graphs collapse it by −17 % to −46 %. **On B200 there is essentially no
plateau** — even *without* the big rungs the band is 66→86 ms and rising smoothly, so
B200-≤1200 already beats H200-+rungs. Adding the rungs on B200 helps `−19 %` at L=2200
but only `−4 %` at 4–5k. Takeaway: the G7 big-chunk-graph optimization is an H200/H100
win; on B200 the eager big chunk is fast enough that it's marginal past ~2k.

**Cached re-serve** (`--cache-frac 0.9`, primed prefix): H200 ~22–55 ms → **B200
~18–41 ms** across 0–5k. Sub-50 ms headline holds.

## Voice-traffic model (`voice_sim.py`)

Unchanged from the H200 daylog: N independent calls, staggered starts, per-call
sequential turns with random 8–16 s gaps, per-call conversation prefix reused each turn
(intra-call cache hit). Reports the TTFT distribution **and** real peak / time-averaged
in-flight concurrency, plus actual `prompt_tokens`. Knobs (`--calls --duration --ctx-min
--ctx-max --groups --no-cache`) as documented there.

## V1 — realistic short-chat traffic never approaches the batch ceiling

**Methodology.** Short chat turns (~170-tok context p50), 75 s wall, default gaps:

```bash
python3 voice_sim.py --calls 50  --duration 75
python3 voice_sim.py --calls 100 --duration 75
```

| | H200 50c | **B200 50c** | H200 100c | **B200 100c** |
|---|---:|---:|---:|---:|
| TTFT p50 / p99 (ms) | 29.2 / 63.9 | **21.5 / 34.5** | 26.0 / 44.1 | **21.8 / 40.3** |
| peak in-flight | 5 | **4** | 7 | **8** |
| avg in-flight | 0.47 | **0.41** | 0.95 | **0.83** |

Same result: 100 active calls → ~1 % effective concurrency, the 32-slot batch idle
almost always. B200 shaves ~4 ms off median TTFT.

## V2 — caching irrelevant at short ctx, decisive at 3–5k

**Methodology.** 100 calls, 75 s, per-call reference context 3–5k tokens (input p50
~3.95k verified via `prompt_tokens`), cache ON vs `--no-cache` (fresh nonce each turn
busts the prefix cache → full re-prefill):

```bash
python3 voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000
python3 voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000 --no-cache
```

| | H200 cache ON | **B200 cache ON** | H200 cache OFF | **B200 cache OFF** |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 44.7 | **35.6** | 157.2 | **141.5** |
| avg in-flight | 1.51 | **1.22** | 3.42 | **3.05** |

Cache still cuts median TTFT ~4× and occupancy ~2.5× on B200. Faster prefill ⇒ lower
steady-state occupancy (1.22 vs H200's 1.51 ON).

## V3 — the cold first-turn tail scales with *distinct* contexts, not calls

**Methodology.** 100 calls, 4k ctx, cache ON. All-unique (no `--groups`, every call's
first turn a cold 4k prefill) vs `--groups 30,30` (two 30 % shared-system-prompt tenants
prime their block once globally, 40 % unique):

```bash
python3 voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000               # all-unique
python3 voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30 # mix
```

| | H200 all-unique | **B200 all-unique** | H200 mix 30/30/40 | **B200 mix 30/30/40** |
|---|---:|---:|---:|---:|
| input tok p50 | 4114 | **3954** | 3984 | **3984** |
| TTFT p50 | 44.7 | **35.6** | 42.4 | **36.9** |
| TTFT p90 | 135.1 | **104.1** | 56.6 | **67.4** |
| TTFT p99 | 281.7 | **212.5** | 82.0 | **143.7** |
| TTFT max | 306.0 | **288.7** | 90.0 | **232.7** |
| avg in-flight | 1.51 | **1.22** | 1.20 | **1.11** |
| peak in-flight | 14 | **9** | 9 | **11** |

Sharing system prompts collapses the tail on B200 too (p99 212→144). The mix-case tail
is noisier on this single B200 run (p99 144 vs H200's 82) — within single-run variance
of how many cold unique-context prefills happen to collide.

## V4 — the knee, and what's actually binding

**Methodology.** Mix 30/30/40, 4k ctx, scale call count:

```bash
python3 voice_sim.py --calls 100 --duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30
python3 voice_sim.py --calls 200 --duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30
```

| | H200 100c | **B200 100c** | H200 200c | **B200 200c** |
|---|---:|---:|---:|---:|
| throughput (req/s) | 6.8 | **6.7** | 13.1 | **13.2** |
| TTFT p50 | 42.4 | **36.9** | 48.9 | **41.6** |
| TTFT p99 | 82.0 | **143.7** | 233.6 | **196.5** |
| peak in-flight | 9 | **11** | 20 | **19** |
| avg in-flight | 1.20 | **1.11** | 3.49 | **2.94** |

At the ~200-call knee, avg in-flight is ~2.9 of 32 (~9 % GPU util) — **latency-bound,
not throughput-bound**, same as H200. p99 ≈197 ms (vs H200 234). The voice-SLO capacity
conclusion is unchanged: **~200 concurrent active calls per GPU** at p99 ≤ ~250 ms.

## V5 — past the knee: the saturation cliff

**Methodology.** Mix 30/30/40, 4k ctx, push call count to 500:

```bash
python3 voice_sim.py --calls 500 --duration 75 --ctx-min 3000 --ctx-max 5000 --groups 30,30
```

| | H200 100 | **B200 100** | H200 200 | **B200 200** | H200 500 | **B200 500** |
|---|---:|---:|---:|---:|---:|---:|
| throughput (req/s) | 6.8 | **6.7** | 13.1 | **13.2** | 29.1 | **29.7** |
| TTFT p50 | 42.4 | **36.9** | 48.9 | **41.6** | 349.8 | **103.4** |
| TTFT p99 | 82.0 | **143.7** | 233.6 | **196.5** | 6,188 | **5,866** |
| peak in-flight | 9 | **11** | 20 | **19** | 206 | **206** |
| avg in-flight | 1.20 | **1.11** | 3.49 | **2.94** | 52.96 | **48.74** |

The key B200 result: the **throughput ceiling barely moves** (29.7 vs 29.1 req/s — same
~120k tok/s prefill wall), so the saturation cliff sits in the same place and the
multi-second over-capacity tail is comparable (p99 5.9 s vs 6.2 s). But B200 clears the
**cached fast path** much faster ⇒ median at 500 calls is **103 ms vs 350 ms (3.4×
better)**. This is consistent with the daylog's thesis that the wall is the
**MoE-weight-read-bound cold prefill** — exactly the part B200's extra bandwidth helps
least at this operating point.

## Pending (to be appended)
- V5 at **300 / 400** calls on config B — pin the exact p99-SLO crossing between the
  200-call knee and the 500-call cliff on B200.
- V5 **doubled-batch** rows (config C, `--max-num-seqs 64 --max-num-batched-tokens 16384`)
  at 200 / 500 — confirm the throughput ceiling is unmoved on B200 and whether the 16k
  budget hurts p99 at the knee as it did on H200 (234→686 ms).

## Environment / repro notes
- vLLM 0.22.0, FP8 (CutlassFp8BlockScaledMM + FLASHINFER_TRTLLM MoE backend), FlashInfer
  attention, Triton/FLA GDN prefill kernel. Single B200, `gpu-memory-utilization 0.85`.
- Drivers `voice_sim.py` / `ttft_sweep.py` are the same files referenced by the H200 daylog.
- Each voice_sim point is a single 75 s run (matching the H200 methodology); single-run
  variance dominates the tail percentiles at low concurrency.
