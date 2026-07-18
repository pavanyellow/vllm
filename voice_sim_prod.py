#!/usr/bin/env python3
"""Production-faithful outbound voice-agent traffic simulator.

Models an outbound dialer fleet of --dialed lines:
  * each line loops: dial (ring 5-15 s) -> 10% connect (--connect-rate) ->
    conversation of ~--turns-per-call turns (jittered) -> hangup -> redial.
    90% no-answer/voicemail: occupies the line, generates NO LLM traffic.
  * context = ONE shared tenant prompt (--tenant-tokens, cached across all calls)
    + a UNIQUE per-caller record (--record-tokens) per connected call
    => steady-state arrival of cold caller-record prefills (call churn).
  * conversation history GROWS turn over turn (no reset), like production.
  * tool turns: with --tool-prob, a turn is TWO LLM round-trips:
    request1 (the "tool call", short) -> synthetic ~--tool-result-tokens tool
    result appended -> request2 (the spoken answer).
  * barge-in: with --cancel-prob the stream is closed after the first tokens
    and the next user turn fires after a short gap.

Reports TTFT and time-to-first-sentence (TTFS ~ first 25 tokens) distributions,
real peak/avg in-flight, connect/turn counts, and actual prompt_tokens.
"""

import argparse
import asyncio
import json
import random
import time

import httpx

SYSTEM = ("You are a helpful phone voice assistant for an outbound campaign. "
          "Keep every reply to one or two short, natural spoken sentences.")
_WORDS = ("account balance transfer payment savings credit limit due date amount "
          "fee transaction history available pending statement schedule customer "
          "service policy interest rate deposit withdrawal routing number branch "
          "mortgage loan refinance escrow premium claim coverage deductible invoice "
          "receipt vendor merchant dispute refund authorize verify confirm pending "
          "review approve decline notice reminder summary detail reference context").split()

UTTERANCES = [
    "Hello? Who is this?",
    "Oh okay, what is this about?",
    "Hmm, can you check my account first?",
    "What were my last few transactions?",
    "And when is my next payment due?",
    "Is there a fee if I pay late?",
    "Alright, can you set up a reminder?",
    "Fine, that works for me.",
    "No, that's everything. Goodbye.",
]

_inflight = 0
_peak = 0
_busy_seconds = 0.0


def make_text(target_tokens, rng):
    n_words = max(1, int(target_tokens / 1.05))
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


async def one_request(client, url, model, messages, max_tokens, cancel_after=None,
                      ttfts=None, ttfss=None, ptoks=None):
    """One streamed chat request. cancel_after=N: close stream after N content
    chunks (barge-in). Returns assistant text (possibly truncated)."""
    global _inflight, _peak, _busy_seconds
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"reasoning_effort": "none"}}
    ttft = None
    ttfs = None
    text = ""
    chunks = 0
    _inflight += 1
    _peak = max(_peak, _inflight)
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
                if obj.get("usage") and ptoks is not None:
                    pt = obj["usage"].get("prompt_tokens")
                    if pt is not None:
                        ptoks.append(pt)
                if obj.get("choices"):
                    delta = obj["choices"][0].get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                        text += delta
                        if ttfs is None and chunks >= 25:
                            ttfs = time.perf_counter() - t0
                        if cancel_after is not None and chunks >= cancel_after:
                            break  # barge-in: close the stream
    except Exception:
        pass
    finally:
        total = time.perf_counter() - t0
        _busy_seconds += total
        _inflight -= 1
    if ttft is not None and ttfts is not None:
        ttfts.append(ttft)
    if ttfs is None and ttft is not None:
        ttfs = time.perf_counter() - t0  # short reply: full response < 25 chunks
    if ttfs is not None and ttfss is not None:
        ttfss.append(ttfs)
    return text


async def run_line(line_id, client, a, deadline, tenant_prompt, stats):
    """One dialer line: dial -> maybe connect -> conversation -> redial."""
    rng = random.Random(0xD1A7 ^ line_id)
    await asyncio.sleep(rng.uniform(0, 10))  # desync line startup
    caller_seq = 0
    while time.perf_counter() < deadline:
        # dial + ring
        await asyncio.sleep(rng.uniform(5, 15))
        stats["dials"] += 1
        if rng.random() > a.connect_rate:
            continue  # no answer / voicemail: no LLM traffic
        # connected: fresh caller record => cold prefill for this call
        stats["connects"] += 1
        caller_seq += 1
        record = make_text(a.record_tokens, random.Random((line_id << 20) | caller_seq))
        messages = [{"role": "system",
                     "content": f"{tenant_prompt}\n\nCaller record:\n{record}"}]
        n_turns = max(2, int(rng.gauss(a.turns_per_call, 1.5)))
        for turn in range(n_turns):
            if time.perf_counter() >= deadline:
                break
            messages.append({"role": "user",
                             "content": UTTERANCES[turn % len(UTTERANCES)]})
            barge = rng.random() < a.cancel_prob
            if not barge and rng.random() < a.tool_prob:
                # tool turn: round-trip 1 (the tool call), tool result, round-trip 2
                text1 = await one_request(client, a.url, a.model, messages, 24,
                                          ttfts=stats["ttft"], ttfss=None,
                                          ptoks=stats["ptok"])
                messages.append({"role": "assistant", "content": text1 or "Checking."})
                messages.append({"role": "user",
                                 "content": "[TOOL RESULT] "
                                 + make_text(a.tool_result_tokens, rng)})
                stats["tool_turns"] += 1
                text = await one_request(client, a.url, a.model, messages,
                                         a.max_tokens, ttfts=stats["ttft"],
                                         ttfss=stats["ttfs"], ptoks=stats["ptok"])
            else:
                cancel = rng.randint(3, 10) if barge else None
                if barge:
                    stats["barge_ins"] += 1
                text = await one_request(client, a.url, a.model, messages,
                                         a.max_tokens, cancel_after=cancel,
                                         ttfts=stats["ttft"], ttfss=stats["ttfs"],
                                         ptoks=stats["ptok"])
            messages.append({"role": "assistant", "content": text or "Okay."})
            stats["turns"] += 1
            await asyncio.sleep(rng.uniform(2, 4) if barge
                                else rng.uniform(a.gap_min, a.gap_max))
        # hangup -> loop redials with a NEW caller


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def main_async(a):
    deadline = time.perf_counter() + a.duration
    tenant_prompt = f"{SYSTEM}\n\nCampaign brief:\n{make_text(a.tenant_tokens, random.Random(7))}"
    stats = {"ttft": [], "ttfs": [], "ptok": [], "dials": 0, "connects": 0,
             "turns": 0, "tool_turns": 0, "barge_ins": 0}
    limits = httpx.Limits(max_connections=a.dialed + 10,
                          max_keepalive_connections=a.dialed + 10)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=None, connect=10.0),
                                 limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            run_line(i, client, a, deadline, tenant_prompt, stats)
            for i in range(a.dialed)
        ])
        wall = time.perf_counter() - t0

    r = stats
    print(f"dialed_lines={a.dialed} connect_rate={a.connect_rate} duration={a.duration}s "
          f"tenant={a.tenant_tokens}tok record={a.record_tokens}tok "
          f"tool_prob={a.tool_prob} cancel_prob={a.cancel_prob} "
          f"gaps={a.gap_min}-{a.gap_max}s turns/call~{a.turns_per_call}")
    print(f"dials={r['dials']}  connects={r['connects']} "
          f"({100*r['connects']/max(1,r['dials']):.0f}%)  turns={r['turns']} "
          f"({r['turns']/wall:.1f} turns/s)  tool_turns={r['tool_turns']}  "
          f"barge_ins={r['barge_ins']}  llm_reqs={len(r['ttft'])}")
    if r["ptok"]:
        print(f"input_tokens p50={pct(r['ptok'],.5):.0f}  "
              f"min={min(r['ptok'])}  max={max(r['ptok'])}")
    t = r["ttft"]
    print(f"TTFT ms  p50={pct(t,.5)*1e3:.1f}  p90={pct(t,.9)*1e3:.1f}  "
          f"p99={pct(t,.99)*1e3:.1f}  max={max(t)*1e3:.1f}" if t else "TTFT: none")
    s = r["ttfs"]
    print(f"TTFS ms  p50={pct(s,.5)*1e3:.1f}  p90={pct(s,.9)*1e3:.1f}  "
          f"p99={pct(s,.99)*1e3:.1f}  (first-sentence ~25 tok)" if s else "TTFS: none")
    print(f"in-flight: PEAK={_peak}  avg(time-weighted)={_busy_seconds/wall:.2f}  "
          f"(dialed lines={a.dialed})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="thinkingmachines/Inkling-NVFP4")
    p.add_argument("--dialed", type=int, default=200, help="concurrent dialer lines")
    p.add_argument("--connect-rate", type=float, default=0.10)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--turns-per-call", type=float, default=6.0)
    p.add_argument("--gap-min", type=float, default=8.0)
    p.add_argument("--gap-max", type=float, default=16.0)
    p.add_argument("--tenant-tokens", type=int, default=3000)
    p.add_argument("--record-tokens", type=int, default=400)
    p.add_argument("--tool-prob", type=float, default=0.5)
    p.add_argument("--tool-result-tokens", type=int, default=200)
    p.add_argument("--cancel-prob", type=float, default=0.15)
    p.add_argument("--max-tokens", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
