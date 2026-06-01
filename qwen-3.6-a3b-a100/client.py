"""Minimal client for the qwen-3.6-a3b-a100 server.

Two modes:
  - one-shot prompt:  python client.py "your prompt here"
  - chat (streaming): python client.py --chat "your prompt here"
  - read prompts from a JSONL/JSON file:  python client.py --input prompts.json

Defaults match the voice-agent scenario: greedy decoding, no thinking preamble.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="prompt text (omit if using --input)")
    p.add_argument("--input", help="JSONL or JSON file of prompts. See README.")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="qwen-3.6-a3b")
    p.add_argument("--chat", action="store_true",
                   help="use chat completions instead of plain completions (applies chat template)")
    p.add_argument("--stream", action="store_true",
                   help="stream the response token-by-token")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--system", default=None, help="optional system prompt (chat mode only)")
    return p.parse_args()


def iter_prompts(args):
    """Yield prompts: from --input file (JSONL with {prompt}/{messages}, or list[str], or list[dict]),
    OR from the positional CLI arg, OR from stdin if neither."""
    if args.input:
        text = Path(args.input).read_text()
        if args.input.endswith(".jsonl"):
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.loads(text)
            if isinstance(data, list):
                yield from data
            else:
                yield data
    elif args.prompt:
        yield args.prompt
    else:
        for line in sys.stdin:
            yield line.rstrip()


def normalize(item, args):
    """Convert anything into (kind, payload) where kind ∈ {'text','messages'}."""
    if isinstance(item, str):
        if args.chat:
            msgs = []
            if args.system:
                msgs.append({"role": "system", "content": args.system})
            msgs.append({"role": "user", "content": item})
            return "messages", msgs
        return "text", item
    if isinstance(item, dict):
        if "messages" in item:
            return "messages", item["messages"]
        if "prompt" in item:
            return ("messages", [{"role": "user", "content": item["prompt"]}]) if args.chat \
                   else ("text", item["prompt"])
    raise ValueError(f"don't know how to handle prompt: {item!r}")


def run_one(client, args, kind, payload, idx):
    """Issue one request, print response, return (ttft_ms, total_ms, n_in, n_out)."""
    t0 = time.perf_counter()
    if kind == "messages":
        if args.stream:
            stream = client.chat.completions.create(
                model=args.model, messages=payload,
                max_tokens=args.max_tokens, temperature=args.temperature,
                stream=True, stream_options={"include_usage": True},
            )
            ttft = None
            chunks = []
            usage = None
            for chunk in stream:
                if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                    ttft = (time.perf_counter() - t0) * 1000
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    chunks.append(delta)
                    print(delta, end="", flush=True)
                if chunk.usage:
                    usage = chunk.usage
            print()
            total = (time.perf_counter() - t0) * 1000
            n_in = usage.prompt_tokens if usage else -1
            n_out = usage.completion_tokens if usage else len("".join(chunks).split())
        else:
            r = client.chat.completions.create(
                model=args.model, messages=payload,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            total = (time.perf_counter() - t0) * 1000
            ttft = total  # without streaming, can't separate TTFT from total
            print(r.choices[0].message.content)
            n_in, n_out = r.usage.prompt_tokens, r.usage.completion_tokens
    else:
        if args.stream:
            stream = client.completions.create(
                model=args.model, prompt=payload,
                max_tokens=args.max_tokens, temperature=args.temperature,
                stream=True, stream_options={"include_usage": True},
            )
            ttft = None
            chunks = []
            usage = None
            for chunk in stream:
                if ttft is None and chunk.choices and chunk.choices[0].text:
                    ttft = (time.perf_counter() - t0) * 1000
                if chunk.choices and chunk.choices[0].text:
                    chunks.append(chunk.choices[0].text)
                    print(chunk.choices[0].text, end="", flush=True)
                if chunk.usage:
                    usage = chunk.usage
            print()
            total = (time.perf_counter() - t0) * 1000
            n_in = usage.prompt_tokens if usage else -1
            n_out = usage.completion_tokens if usage else len("".join(chunks).split())
        else:
            r = client.completions.create(
                model=args.model, prompt=payload,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            total = (time.perf_counter() - t0) * 1000
            ttft = total
            print(r.choices[0].text)
            n_in, n_out = r.usage.prompt_tokens, r.usage.completion_tokens

    print(f"\n[#{idx}] in={n_in}tok out={n_out}tok ttft={ttft:.0f}ms total={total:.0f}ms",
          file=sys.stderr)
    return ttft, total, n_in, n_out


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    for i, item in enumerate(iter_prompts(args)):
        kind, payload = normalize(item, args)
        run_one(client, args, kind, payload, i)


if __name__ == "__main__":
    main()
