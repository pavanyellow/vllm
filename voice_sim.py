#!/usr/bin/env python3
"""Realistic voice-traffic simulator for a chat-completions endpoint.

Models N independent phone calls. Each call is *sequential* (one LLM request at
a time, never overlapping itself); between turns there is a random human-speak +
TTS-playback + silence gap (default 8-16 s). Calls start at staggered offsets so
they desync. Within a call the conversation history is reused verbatim each turn
=> prefix-cache HIT on the shared prefix, fresh prefill only on the new turn.
Each call is tagged with a unique id so calls don't share cache with each other.

It measures what actually matters for capacity:
  * the TTFT distribution across all turns (p50/p90/p99/max), and
  * the REAL peak & time-averaged number of requests in-flight at once
    (send -> full-completion), which is far below the active-call count when
    each call is only "busy" ~200-400 ms out of every ~8-16 s.
"""

import argparse
import asyncio
import json
import random
import time

import httpx

SYSTEM = ("You are a helpful phone voice assistant. Keep every reply to one or "
          "two short, natural spoken sentences.")
_WORDS = ("account balance transfer payment savings credit limit due date amount "
          "fee transaction history available pending statement schedule customer "
          "service policy interest rate deposit withdrawal routing number branch "
          "mortgage loan refinance escrow premium claim coverage deductible invoice "
          "receipt vendor merchant dispute refund authorize verify confirm pending "
          "review approve decline notice reminder summary detail reference context").split()


def make_context(target_tokens, rng):
    # ~1.4 tokens/word for this tokenizer; build a fixed filler "reference doc".
    n_words = max(1, int(target_tokens / 1.05))
    return "Reference context: " + " ".join(rng.choice(_WORDS) for _ in range(n_words))
UTTERANCES = [
    "Hi, can you help me check my account balance?",
    "What were my last few transactions?",
    "Okay, and when is my next payment due?",
    "Can you move five hundred to my savings?",
    "Actually, make that three hundred instead.",
    "Got it. Is there a fee for that transfer?",
    "Thanks. What's my available credit right now?",
    "Can you read me that last one more time?",
    "Alright, anything else I should know about?",
    "Perfect, that's all for today. Goodbye.",
]

# in-flight tracking (single asyncio thread -> no lock needed)
_inflight = 0
_peak = 0
_busy_seconds = 0.0
_inflight_samples = []   # concurrency seen at each request's send time


async def one_turn(client, url, model, messages, max_tokens):
    """Send one streamed chat request; return (ttft_s, total_s, assistant_text)."""
    global _inflight, _peak, _busy_seconds
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    ttft = None
    text = ""
    prompt_tokens = None
    _inflight += 1
    _peak = max(_peak, _inflight)
    _inflight_samples.append(_inflight)
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("usage"):
                    prompt_tokens = obj["usage"].get("prompt_tokens")
                if obj.get("choices"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    delta = obj["choices"][0]["delta"].get("content")
                    if delta:
                        text += delta
    finally:
        total = time.perf_counter() - t0
        _busy_seconds += total
        _inflight -= 1
    return ttft, total, text, prompt_tokens


async def run_call(call_id, client, url, model, deadline, gap_min, gap_max,
                   max_tokens, results, ptoks, no_cache, ctx_on, sys_content):
    # stagger start so calls don't align
    await asyncio.sleep(random.uniform(0, gap_max))
    # sys_content is precomputed: a SHARED block (identical across a group =>
    # cross-call cache hit) or a per-call UNIQUE block (cold first turn).
    messages = [{"role": "system", "content": sys_content}]
    turn = 0
    while time.perf_counter() < deadline:
        if no_cache:
            # fresh nonce at the FRONT each turn => first-block hash changes =>
            # no prefix-cache hit, full context re-prefills every turn.
            messages[0] = {"role": "system",
                           "content": f"[n{random.getrandbits(48)}] {sys_content}"}
        if ctx_on:
            # large-context mode: hold input length stable (context + current
            # turn only) instead of letting history growth drift it.
            del messages[1:]
        messages.append({"role": "user", "content": UTTERANCES[turn % len(UTTERANCES)]})
        ttft, total, text, pt = await one_turn(client, url, model, messages, max_tokens)
        if ttft is not None:
            results.append(ttft)
        if pt is not None:
            ptoks.append(pt)
        messages.append({"role": "assistant", "content": text or "Okay."})
        turn += 1
        await asyncio.sleep(random.uniform(gap_min, gap_max))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def main_async(a):
    deadline = time.perf_counter() + a.duration
    results = []
    ptoks = []
    ctx_on = a.ctx_max > 0

    def ctx_for(seed):
        if not ctx_on:
            return ""
        return "\n\n" + make_context(random.Random(seed).uniform(a.ctx_min, a.ctx_max),
                                     random.Random(seed))

    # Assign each call a system prompt. --groups "30,30" => two SHARED groups of
    # 30% (identical content within a group), remaining 40% UNIQUE per call.
    group_pcts = [float(x) for x in a.groups.split(",")] if a.groups else []
    shared_sys = [f"{SYSTEM} (group {g}){ctx_for(0x5ADE5 ^ g)}"
                  for g in range(len(group_pcts))]
    assign = []
    for g, gp in enumerate(group_pcts):
        assign.extend([shared_sys[g]] * round(a.calls * gp / 100.0))
    assign = assign[:a.calls]
    while len(assign) < a.calls:
        cid = len(assign)
        assign.append(f"{SYSTEM} (session {cid}){ctx_for(0xC0FFEE ^ cid)}")
    n_shared = sum(round(a.calls * p / 100.0) for p in group_pcts)

    limits = httpx.Limits(max_connections=a.calls + 10, max_keepalive_connections=a.calls + 10)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=None, connect=10.0),
                                 limits=limits) as client:
        t_start = time.perf_counter()
        await asyncio.gather(*[
            run_call(i, client, a.url, a.model, deadline, a.gap_min, a.gap_max,
                     a.max_tokens, results, ptoks, a.no_cache, ctx_on, assign[i])
            for i in range(a.calls)
        ])
        wall = time.perf_counter() - t_start

    ptinfo = (f"  input_tokens p50={pct(ptoks,.50):.0f} min={min(ptoks)} max={max(ptoks)}"
              if ptoks else "")
    mix = (f" groups={a.groups} (shared={n_shared} unique={a.calls - n_shared})"
           if group_pcts else "")
    print(f"calls={a.calls} duration={a.duration}s gap={a.gap_min}-{a.gap_max}s "
          f"max_tokens={a.max_tokens} cache={'OFF' if a.no_cache else 'ON'} "
          f"ctx={a.ctx_min:.0f}-{a.ctx_max:.0f}{mix}")
    print(f"turns completed : {len(results)}  ({len(results)/wall:.1f} req/s avg){ptinfo}")
    print(f"TTFT ms  p50={pct(results,.50)*1e3:.1f}  p90={pct(results,.90)*1e3:.1f}  "
          f"p99={pct(results,.99)*1e3:.1f}  max={max(results)*1e3:.1f}")
    print(f"in-flight: PEAK={_peak}  avg(time-weighted)={_busy_seconds/wall:.2f}  "
          f"(active calls={a.calls})")
    # concurrency-at-send histogram
    from collections import Counter
    hist = Counter(_inflight_samples)
    dist = "  ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
    print(f"requests by #in-flight-at-send : {dist}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--duration", type=float, default=75.0, help="wall-clock seconds")
    p.add_argument("--gap-min", type=float, default=8.0)
    p.add_argument("--gap-max", type=float, default=16.0)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--no-cache", action="store_true",
                   help="bust prefix cache each turn => full context re-prefill")
    p.add_argument("--ctx-min", type=float, default=0.0,
                   help="min per-call reference-context target tokens (0 = short mode)")
    p.add_argument("--ctx-max", type=float, default=0.0,
                   help="max per-call reference-context target tokens")
    p.add_argument("--groups", default="",
                   help="comma %% of calls sharing each system prompt, e.g. '30,30' "
                        "=> two shared groups, remainder unique per call")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
