"""Generate a JSONL sweep file with prompt_token_ids at controlled lengths.

Lengths: 1-10 (every 1), 20-100 (every 10), 200-6000 (every 100) = 78 points.
Each length gets N_PROMPTS distinct random-token prompts for statistical robustness.

Output: sweep_prompts.jsonl — feed directly to benchmark.py
"""
import json
import random
import argparse
from pathlib import Path

VOCAB_LO = 1000
VOCAB_HI = 240000

def make_lengths():
    lengths = list(range(1, 11))
    lengths += list(range(20, 101, 10))
    lengths += list(range(200, 6001, 100))
    return lengths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="sweep_prompts.jsonl")
    p.add_argument("--prompts-per-length", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    lengths = make_lengths()
    out = Path(args.out)

    count = 0
    with out.open("w") as f:
        for length in lengths:
            for i in range(args.prompts_per_length):
                ids = [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(length)]
                f.write(json.dumps({"prompt_token_ids": ids}) + "\n")
                count += 1

    print(f"wrote {count} prompts ({len(lengths)} lengths x {args.prompts_per_length}) to {out}")

if __name__ == "__main__":
    main()
