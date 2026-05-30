#!/usr/bin/env python3
"""Thinking vs Instruct mode on a chat endpoint (Qwen3.x reasoning parser).

Hits /v1/chat/completions streaming, once per mode:
  - thinking (default): no extra kwargs
  - instruct: chat_template_kwargs={"enable_thinking": false}
Measures TTFT, decode ITL, tokens generated, wall time, and how the qwen3
reasoning parser splits reasoning_content vs content in the stream.
"""
import argparse, json, time
import requests

PROMPT = ("A snail climbs a 12-meter pole. Each day it climbs 3 meters, "
          "each night it slides back 2 meters. On which day does it reach the top? "
          "Give the day number.")


def run(url, model, enable_thinking, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    t0 = time.perf_counter()
    ttft = None
    t_first_content = None
    chunk_times = []
    reasoning_chars = 0
    content_chars = 0
    usage = None
    with requests.post(url, json=body, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode() if isinstance(line, bytes) else line
            if not s.startswith("data:"):
                continue
            data = s[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta", {})
            now = time.perf_counter()
            rc = delta.get("reasoning_content")
            cc = delta.get("content")
            if rc:
                if ttft is None:
                    ttft = now - t0
                reasoning_chars += len(rc)
                chunk_times.append(now)
            if cc:
                if ttft is None:
                    ttft = now - t0
                if t_first_content is None:
                    t_first_content = now - t0
                content_chars += len(cc)
                chunk_times.append(now)
    wall = time.perf_counter() - t0
    n_tok = usage.get("completion_tokens") if usage else len(chunk_times)
    itl = ((chunk_times[-1] - chunk_times[0]) / (len(chunk_times) - 1) * 1000
           if len(chunk_times) > 1 else float("nan"))
    return {
        "ttft_ms": ttft * 1000 if ttft else float("nan"),
        "first_content_ms": t_first_content * 1000 if t_first_content else float("nan"),
        "itl_ms": itl,
        "out_tokens": n_tok,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "wall_ms": wall * 1000,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000/v1/chat/completions")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    print(f"prompt: {PROMPT!r}\n")
    hdr = f"{'mode':>9} {'TTFT':>7} {'1stCont':>8} {'ITL':>6} {'out_tok':>8} {'reason_ch':>10} {'cont_ch':>8} {'wall_ms':>9}"
    for mode, et in [("thinking", None), ("instruct", False)]:
        rows = [run(a.url, a.model, et, a.max_tokens) for _ in range(a.reps)]
        # median by wall
        rows.sort(key=lambda r: r["wall_ms"])
        m = rows[len(rows)//2]
        if mode == "thinking":
            print(hdr)
        print(f"{mode:>9} {m['ttft_ms']:>7.1f} {m['first_content_ms']:>8.1f} "
              f"{m['itl_ms']:>6.2f} {m['out_tokens']:>8} {m['reasoning_chars']:>10} "
              f"{m['content_chars']:>8} {m['wall_ms']:>9.1f}")


if __name__ == "__main__":
    main()
