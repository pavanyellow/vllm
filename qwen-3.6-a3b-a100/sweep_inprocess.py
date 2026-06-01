"""In-process TTFT sweep — no HTTP overhead.

Loads vLLM directly, generates random-token prompts at 78 lengths (1-6000),
measures raw TTFT per call. Outputs per_call.csv compatible with make_sweep_report.py.

Usage:
  python sweep_inprocess.py --out /workspace/sweep_inprocess
"""
import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch


def make_lengths():
    lengths = list(range(1, 11))
    lengths += list(range(20, 101, 10))
    lengths += list(range(200, 6001, 100))
    return lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", default="/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8")
    p.add_argument("--prompts-per-length", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=6300)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=4)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    lengths = make_lengths()

    # Build extended CUDA graph capture sizes (same as start_server.sh)
    default_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    extras = [640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    capture_sizes = sorted(set(default_sizes + extras))

    print(f"[sweep] loading vLLM...", flush=True)
    t0 = time.time()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=False,
        dtype="auto",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": capture_sizes[-1],
        },
    )
    print(f"[sweep] loaded in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    gpu_name = torch.cuda.get_device_name(0)

    # Global warmup at max length
    print(f"[sweep] warmup (5 calls at L={max(lengths)})...", flush=True)
    warm_ids = [rng.randint(1000, 240000) for _ in range(max(lengths))]
    for _ in range(5):
        llm.generate([{"prompt_token_ids": warm_ids}], sp, use_tqdm=False)

    records = []
    for length in lengths:
        print(f"[sweep] L={length:>5}", end="", flush=True)
        ttfts = []
        for i in range(args.prompts_per_length):
            ids = [rng.randint(1000, 240000) for _ in range(length)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_obj = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t0) * 1000
            n_in = length
            ttfts.append(ttft_ms)
            records.append({
                "input_idx": len(records),
                "trial": 0,
                "ttft_ms": round(ttft_ms, 2),
                "total_ms": round(ttft_ms, 2),
                "n_in": length,
                "n_out": 1,
            })
        med = statistics.median(ttfts)
        print(f"  p50={med:>7.1f}ms  all=[{', '.join(f'{t:.0f}' for t in ttfts)}]", flush=True)

    # Write per_call.csv (compatible with make_sweep_report.py)
    csv_path = out / "per_call.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input_idx", "trial", "ttft_ms", "total_ms", "n_in", "n_out"])
        w.writeheader()
        w.writerows(records)

    # Write metadata
    meta_path = out / "sweep_meta.json"
    meta_path.write_text(json.dumps({
        "gpu": gpu_name,
        "model": args.model_dir,
        "max_model_len": args.max_model_len,
        "prompts_per_length": args.prompts_per_length,
        "n_lengths": len(lengths),
        "n_records": len(records),
        "group_size_m": "64 (manual override)",
    }, indent=2))

    print(f"\n[sweep] done — {len(records)} measurements across {len(lengths)} lengths")
    print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] run: python make_sweep_report.py --input {csv_path} --out {out}")


if __name__ == "__main__":
    main()
