#!/usr/bin/env python3
"""Realistic voice-agent traffic with a controlled per-request cache hit rate.

N concurrent call slots. Each slot runs back-to-back calls (caller hangs up after
6-10 turns, a new caller takes the slot), so concurrency stays fixed while callers
churn. Shape of every request, matching how production voice agents build prompts:

  system  = tenant prompt (~3k tok, shared by every call on that agent)
            + caller block at the END (per-call variables: name, account, notes)
  history = grows turn by turn: caller utterance + a fresh context injection
            (tool result / retrieved notes) + the assistant's previous reply

The fresh part of each request is sized to ~(1-hit)/hit of the cached prefix, so
the MEDIAN request is ~`--target-hit` cached (default 90%). Per-request cached
tokens are read from the response usage and reported as a distribution, so the
hit rate is measured, not assumed. A random per-run salt leads every prompt, so
no run reuses another run's cache.
"""

import argparse
import asyncio
import json
import random
import time

import httpx

_WORDS = ("account balance transfer payment savings credit limit due date amount "
          "fee transaction history available pending statement schedule customer "
          "service policy interest rate deposit withdrawal routing number branch "
          "mortgage loan refinance escrow premium claim coverage deductible invoice "
          "receipt vendor merchant dispute refund authorize verify confirm pending "
          "review approve decline notice reminder summary detail reference context").split()
UTTERANCES = [
    "Hi, can you help me check my account balance?", "What were my last few transactions?",
    "Okay, and when is my next payment due?", "Can you move five hundred to my savings?",
    "Actually, make that three hundred instead.", "Got it. Is there a fee for that transfer?",
    "Thanks. What's my available credit right now?", "Can you read me that last one more time?",
    "Alright, anything else I should know about?", "Perfect, that's all for today. Goodbye.",
]
TOK_PER_WORD = 1.05  # calibrate with --calibrate; reported prompt_tokens is the truth


def text(n_tokens, rng):
    return " ".join(rng.choice(_WORDS) for _ in range(max(1, int(n_tokens / TOK_PER_WORD))))


_inflight = 0
_peak = 0
_busy = 0.0
_errors = {}


async def request(client, url, model, messages, max_tokens):
    """One streamed turn -> (ttft_s or None, reply_text, prompt_tokens, cached_tokens)."""
    global _inflight, _peak, _busy
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}}
    ttft, reply, pt, ct = None, "", None, None
    _inflight += 1
    _peak = max(_peak, _inflight)
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                _errors[r.status_code] = _errors.get(r.status_code, 0) + 1
                await r.aread()
                return None, "", None, None
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
                    pt = obj["usage"].get("prompt_tokens")
                    ct = (obj["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ttft is None and (d.get("content") or d.get("tool_calls")):
                        ttft = time.perf_counter() - t0
                    reply += d.get("content") or ""
        if ttft is None:
            _errors["no_token"] = _errors.get("no_token", 0) + 1
    except Exception as e:
        _errors[type(e).__name__] = _errors.get(type(e).__name__, 0) + 1
    finally:
        _busy += time.perf_counter() - t0
        _inflight -= 1
    return ttft, reply, pt, ct


async def slot(sid, client, a, deadline, tenants, stats, salt):
    rng = random.Random(f"{salt}:{sid}")
    await asyncio.sleep(rng.uniform(0, a.gap_max))  # desync slots
    frac = (1 - a.target_hit) / a.target_hit       # fresh tokens per cached token
    call_no = 0
    while time.perf_counter() < deadline:
        call_no += 1
        tenant = tenants[sid % len(tenants)]
        # first turn: tenant prompt is the shared (cached) prefix, caller block is fresh
        caller = text(frac * a.tenant_tokens, rng)
        messages = [{"role": "system", "content": f"{tenant}\n\nCaller {sid}-{call_no}:\n{caller}"}]
        prefix_est = a.tenant_tokens + frac * a.tenant_tokens  # tokens cached on the next turn
        stats["calls"] += 1
        for turn in range(rng.randint(a.turns_min, a.turns_max)):
            if time.perf_counter() >= deadline:
                return
            if turn > 0:
                # fresh per turn ~= frac x everything already in the prompt
                inj = text(max(0, frac * prefix_est - 40), rng)
                messages.append({"role": "user",
                                 "content": f"{UTTERANCES[turn % len(UTTERANCES)]}\n[context] {inj}"})
            else:
                messages.append({"role": "user", "content": UTTERANCES[0]})
            ttft, reply, pt, ct = await request(client, a.url, a.model, messages, a.max_tokens)
            if ttft is not None:
                stats["ttft"].append(ttft)
                stats["first" if turn == 0 else "later"].append(ttft)
            if pt and ct is not None:
                stats["hit"].append(ct / pt)
                stats["ptok"].append(pt)
            messages.append({"role": "assistant", "content": reply or "Okay."})
            prefix_est = pt or prefix_est * (1 + frac)
            await asyncio.sleep(rng.uniform(a.gap_min, a.gap_max))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def main_async(a):
    salt = a.salt or f"{random.getrandbits(32):08x}"
    trng = random.Random(salt)
    tenants = [f"[run {salt}] You are a phone voice assistant for tenant {t}. Keep every reply to "
               f"one or two short spoken sentences.\n\nPolicies:\n{text(a.tenant_tokens, trng)}"
               for t in range(a.tenants)]
    stats = {"ttft": [], "first": [], "later": [], "hit": [], "ptok": [], "calls": 0}
    deadline = time.perf_counter() + a.duration
    lim = httpx.Limits(max_connections=a.calls + 10, max_keepalive_connections=a.calls + 10)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=None, connect=10.0), limits=lim) as c:
        t0 = time.perf_counter()
        await asyncio.gather(*[slot(i, c, a, deadline, tenants, stats, salt) for i in range(a.calls)])
        wall = time.perf_counter() - t0
    t, h = stats["ttft"], stats["hit"]
    sent = len(t) + sum(_errors.values())
    print(f"calls={a.calls} duration={a.duration:.0f}s tenants={a.tenants} tenant_tok={a.tenant_tokens} "
          f"target_hit={a.target_hit:.0%} max_tokens={a.max_tokens} gap={a.gap_min}-{a.gap_max}s salt={salt}")
    print(f"requests={sent} ({len(t)/wall:.1f}/s ok)  callers={stats['calls']}  "
          f"errors={sum(_errors.values())} {dict(_errors) if _errors else ''}")
    if h:
        print(f"per-request cache hit: p10={pct(h,.1):.1%} median={pct(h,.5):.1%} p90={pct(h,.9):.1%}  "
              f"prompt tokens median={pct(stats['ptok'],.5):.0f}")
    if t:
        print(f"TTFT ms  p50={pct(t,.5)*1e3:.0f}  p90={pct(t,.9)*1e3:.0f}  p99={pct(t,.99)*1e3:.0f}  "
              f"max={max(t)*1e3:.0f}   first-turn p50={pct(stats['first'],.5)*1e3:.0f}  "
              f"later-turn p50={pct(stats['later'],.5)*1e3:.0f}")
    print(f"in-flight: peak={_peak} avg={_busy/wall:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:10100/v1/chat/completions")
    p.add_argument("--model", default="gemma-4-31b")
    p.add_argument("--calls", type=int, default=50, help="concurrent call slots")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--tenants", type=int, default=4, help="distinct agent system prompts")
    p.add_argument("--tenant-tokens", type=float, default=3000)
    p.add_argument("--target-hit", type=float, default=0.90, help="per-request cached share")
    p.add_argument("--turns-min", type=int, default=6)
    p.add_argument("--turns-max", type=int, default=10)
    p.add_argument("--gap-min", type=float, default=8.0)
    p.add_argument("--gap-max", type=float, default=16.0)
    p.add_argument("--max-tokens", type=int, default=25)
    p.add_argument("--salt", default="")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
