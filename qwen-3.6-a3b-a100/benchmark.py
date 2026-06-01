"""Benchmark the qwen-3.6-a3b-a100 server end-to-end.

Reads prompts from an input file, fires them at the server one-by-one (single-stream
voice scenario), measures per-request TTFT and total latency, samples GPU
utilization in parallel, writes per-call CSV + aggregate JSON + console summary.

INPUT FILE STRUCTURE
────────────────────
Supported formats (auto-detected by extension):

  *.jsonl  — one JSON object per line, each with one of:
              {"prompt": "..."}                      ← plain completion
              {"prompt": "...", "max_tokens": 256}   ← optional per-request override
              {"messages": [{"role":"user","content":"..."}], "system": "..."}
              {"prompt_token_ids": [123, 456, ...]}  ← exact token IDs (most reproducible)

  *.json   — JSON list of the above objects, OR list of strings.

  *.txt    — one prompt per line (text only).

GLOBAL SETTINGS
───────────────
  --warmup N     issue N requests first and discard timings (default 3)
  --trials N     run each input N times back-to-back (default 1)
  --max-tokens   default 1 (TTFT-only). Bump to 256+ for steady-state TPOT measurement.
  --temperature  default 0.0 (greedy)
  --stream       use SSE streaming. Required to measure true TTFT (otherwise total==TTFT).
  --chat         use /v1/chat/completions (applies the chat template). Default off.

OUTPUTS
───────
  <out>/per_call.csv         per-request: input_idx, trial, ttft_ms, total_ms, n_in, n_out
  <out>/summary.json         aggregate: count, p50/p90/p99 ttft, GPU avg util, etc.
  <out>/gpu_samples.csv      raw nvidia-smi samples taken during the bench

Usage examples
──────────────
  # quick smoke
  python benchmark.py --input my_prompts.jsonl --out bench_out

  # voice-agent simulation (single-stream, streaming TTFT, short outputs)
  python benchmark.py --input voice_prompts.jsonl --stream --max-tokens 1 \
      --warmup 5 --trials 5 --out bench_voice

  # measure full generation latency
  python benchmark.py --input chat_prompts.jsonl --chat --stream \
      --max-tokens 256 --out bench_full
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI


# ─── input parsing ────────────────────────────────────────────────────────

def load_inputs(path: str) -> list[dict]:
    """Return a list of dicts, each describing one prompt."""
    p = Path(path)
    if p.suffix == ".jsonl":
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(_normalize_item(json.loads(line)))
        return out
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        return [_normalize_item(x) for x in (data if isinstance(data, list) else [data])]
    if p.suffix == ".txt":
        return [{"prompt": line.strip()} for line in p.read_text().splitlines() if line.strip()]
    raise ValueError(f"unknown input extension: {p.suffix}")


def _normalize_item(item) -> dict:
    if isinstance(item, str):
        return {"prompt": item}
    if isinstance(item, dict):
        if "messages" in item or "prompt" in item or "prompt_token_ids" in item:
            return item
    raise ValueError(f"unrecognized input item: {item!r}")


# ─── GPU sampler ──────────────────────────────────────────────────────────

class GpuSampler:
    """Sample nvidia-smi at fixed interval into a list. Use as context manager."""
    def __init__(self, interval_ms: int = 200, gpu_id: int = 0):
        self.interval = interval_ms / 1000
        self.gpu_id = gpu_id
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                util_gpu, util_mem, mem_used, mem_free, temp, power = [x.strip() for x in out.split(",")]
                self.samples.append({
                    "t_ms": int((time.time() % 86400) * 1000),
                    "util_gpu": float(util_gpu),
                    "util_mem": float(util_mem),
                    "mem_used_mib": float(mem_used),
                    "mem_free_mib": float(mem_free),
                    "temp_c": float(temp),
                    "power_w": float(power) if power.replace(".", "").isdigit() else None,
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> dict:
        if not self.samples:
            return {"n_samples": 0}
        utils = [s["util_gpu"] for s in self.samples]
        mems = [s["mem_used_mib"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "gpu_util_avg": round(sum(utils) / len(utils), 1),
            "gpu_util_p50": round(statistics.median(utils), 1),
            "gpu_util_p99": round(sorted(utils)[int(0.99 * len(utils))], 1) if len(utils) > 1 else utils[0],
            "gpu_util_max": max(utils),
            "mem_used_avg_mib": round(sum(mems) / len(mems), 1),
            "mem_used_max_mib": max(mems),
        }


# ─── one request ──────────────────────────────────────────────────────────

def send_one(client: OpenAI, model: str, item: dict, *, chat: bool, stream: bool,
             max_tokens: int, temperature: float) -> dict:
    """Return per-call metrics dict."""
    t0 = time.perf_counter()
    ttft_ms = None
    n_in = n_out = -1

    if "messages" in item:
        kind = "messages"
        payload = item["messages"]
    elif "prompt_token_ids" in item:
        # vLLM's OpenAI server accepts token-id lists directly via /v1/completions
        kind = "token_ids"
        payload = item["prompt_token_ids"]
    else:
        kind = "text"
        payload = item["prompt"]

    mt = item.get("max_tokens", max_tokens)
    tmp = item.get("temperature", temperature)

    if chat or kind == "messages":
        # Promote text to a single-user-turn message
        if kind != "messages":
            payload = [{"role": "user", "content": payload if kind == "text" else "(see token ids)"}]
        if stream:
            resp = client.chat.completions.create(
                model=model, messages=payload,
                max_tokens=mt, temperature=tmp,
                stream=True, stream_options={"include_usage": True},
            )
            usage = None
            chunks = []
            for c in resp:
                if c.choices and c.choices[0].delta.content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(c.choices[0].delta.content)
                if c.usage:
                    usage = c.usage
            total_ms = (time.perf_counter() - t0) * 1000
            if usage:
                n_in, n_out = usage.prompt_tokens, usage.completion_tokens
        else:
            r = client.chat.completions.create(
                model=model, messages=payload, max_tokens=mt, temperature=tmp,
            )
            total_ms = (time.perf_counter() - t0) * 1000
            ttft_ms = total_ms
            n_in, n_out = r.usage.prompt_tokens, r.usage.completion_tokens
    else:
        if stream:
            resp = client.completions.create(
                model=model, prompt=payload,
                max_tokens=mt, temperature=tmp,
                stream=True, stream_options={"include_usage": True},
            )
            usage = None
            for c in resp:
                if c.choices and c.choices[0].text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                if c.usage:
                    usage = c.usage
            total_ms = (time.perf_counter() - t0) * 1000
            if usage:
                n_in, n_out = usage.prompt_tokens, usage.completion_tokens
        else:
            r = client.completions.create(
                model=model, prompt=payload, max_tokens=mt, temperature=tmp,
            )
            total_ms = (time.perf_counter() - t0) * 1000
            ttft_ms = total_ms
            n_in, n_out = r.usage.prompt_tokens, r.usage.completion_tokens

    return {"ttft_ms": round(ttft_ms or total_ms, 2),
            "total_ms": round(total_ms, 2),
            "n_in": n_in, "n_out": n_out}


# ─── main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="prompts file (.jsonl, .json, .txt)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="qwen-3.6-a3b")
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--stream", action="store_true", help="use SSE streaming (required for true TTFT)")
    p.add_argument("--chat", action="store_true", help="use chat-completions API")
    p.add_argument("--gpu-sample-ms", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args.input)
    print(f"[bench] loaded {len(inputs)} inputs from {args.input}", file=sys.stderr)
    print(f"[bench] {args.warmup} warmup + {args.trials} trial(s) per input "
          f"= {args.warmup + len(inputs)*args.trials} total requests",
          file=sys.stderr)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Warm-up — discard timings
    for i in range(args.warmup):
        try:
            send_one(client, args.model, inputs[i % len(inputs)],
                     chat=args.chat, stream=args.stream,
                     max_tokens=args.max_tokens, temperature=args.temperature)
        except Exception as e:
            print(f"[warmup] #{i} failed: {e}", file=sys.stderr)
    print(f"[bench] warmup done", file=sys.stderr)

    # Real bench, GPU-sampled
    records: list[dict] = []
    with GpuSampler(interval_ms=args.gpu_sample_ms) as gpu:
        for trial in range(args.trials):
            for idx, item in enumerate(inputs):
                try:
                    m = send_one(client, args.model, item,
                                 chat=args.chat, stream=args.stream,
                                 max_tokens=args.max_tokens, temperature=args.temperature)
                    rec = {"trial": trial, "idx": idx, **m}
                    records.append(rec)
                    print(f"[bench] t{trial} #{idx:>3}: in={rec['n_in']:>5}tok "
                          f"out={rec['n_out']:>4}tok ttft={rec['ttft_ms']:>7.1f}ms "
                          f"total={rec['total_ms']:>7.1f}ms",
                          file=sys.stderr)
                except Exception as e:
                    print(f"[bench] t{trial} #{idx} FAILED: {e}", file=sys.stderr)

    # Write per-call CSV
    if records:
        with (out / "per_call.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    # Write GPU samples
    with (out / "gpu_samples.csv").open("w") as f:
        if gpu.samples:
            w = csv.DictWriter(f, fieldnames=list(gpu.samples[0].keys()))
            w.writeheader()
            w.writerows(gpu.samples)

    # Aggregate
    if records:
        ttfts = sorted(r["ttft_ms"] for r in records)
        totals = sorted(r["total_ms"] for r in records)
        n_outs = [r["n_out"] for r in records]
        n_ins = [r["n_in"] for r in records]
        def q(arr, p): return arr[min(int(p * len(arr)), len(arr) - 1)]
        agg = {
            "n_requests": len(records),
            "n_unique_inputs": len(inputs),
            "trials_per_input": args.trials,
            "ttft_ms": {
                "min":  round(ttfts[0], 1),
                "p50":  round(q(ttfts, 0.50), 1),
                "p90":  round(q(ttfts, 0.90), 1),
                "p99":  round(q(ttfts, 0.99), 1),
                "max":  round(ttfts[-1], 1),
                "mean": round(sum(ttfts) / len(ttfts), 1),
            },
            "total_ms": {
                "min":  round(totals[0], 1),
                "p50":  round(q(totals, 0.50), 1),
                "p99":  round(q(totals, 0.99), 1),
                "max":  round(totals[-1], 1),
            },
            "tokens": {
                "input_min":  min(n_ins),
                "input_p50":  q(sorted(n_ins), 0.50),
                "input_max":  max(n_ins),
                "output_min": min(n_outs),
                "output_p50": q(sorted(n_outs), 0.50),
                "output_max": max(n_outs),
            },
            "tpot_ms_p50": (
                round(statistics.median([(r["total_ms"] - r["ttft_ms"]) / max(r["n_out"] - 1, 1)
                                         for r in records if r["n_out"] > 1]), 2)
                if any(r["n_out"] > 1 for r in records) else None
            ),
            "gpu": gpu.summary(),
            "config": {
                "model": args.model, "base_url": args.base_url,
                "max_tokens": args.max_tokens, "temperature": args.temperature,
                "stream": args.stream, "chat": args.chat,
                "warmup": args.warmup, "trials": args.trials,
            },
        }
        (out / "summary.json").write_text(json.dumps(agg, indent=2))

        print()
        print("=" * 62, file=sys.stderr)
        print(f" BENCHMARK SUMMARY  ({agg['n_requests']} requests)",
              file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        print(f"  TTFT  min={agg['ttft_ms']['min']:>6.1f}  p50={agg['ttft_ms']['p50']:>6.1f}  "
              f"p90={agg['ttft_ms']['p90']:>6.1f}  p99={agg['ttft_ms']['p99']:>6.1f}  "
              f"max={agg['ttft_ms']['max']:>6.1f}  (ms)",
              file=sys.stderr)
        print(f"  TOT   min={agg['total_ms']['min']:>6.1f}  p50={agg['total_ms']['p50']:>6.1f}  "
              f"p99={agg['total_ms']['p99']:>6.1f}  max={agg['total_ms']['max']:>6.1f}  (ms)",
              file=sys.stderr)
        if agg["tpot_ms_p50"] is not None:
            print(f"  TPOT  p50={agg['tpot_ms_p50']:.2f} ms/tok", file=sys.stderr)
        print(f"  IN tokens : p50={agg['tokens']['input_p50']}  max={agg['tokens']['input_max']}",
              file=sys.stderr)
        print(f"  OUT tokens: p50={agg['tokens']['output_p50']}  max={agg['tokens']['output_max']}",
              file=sys.stderr)
        g = agg["gpu"]
        print(f"  GPU   util avg={g.get('gpu_util_avg', '-')}%  p50={g.get('gpu_util_p50', '-')}%  "
              f"max={g.get('gpu_util_max', '-')}%  mem_max={g.get('mem_used_max_mib', '-')} MiB",
              file=sys.stderr)
        print()
        print(f"  outputs: {out}/{{per_call.csv,gpu_samples.csv,summary.json}}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
