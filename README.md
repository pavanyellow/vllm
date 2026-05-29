# Serving Qwen/Qwen3.6-35B-A3B-FP8 on a single H100

This fork documents how we brought up **`Qwen/Qwen3.6-35B-A3B-FP8`** as a low-latency,
text-only ("voice backend") OpenAI-compatible endpoint using vLLM built from this
source checkout, on a single **NVIDIA H100 80GB**.

> Built & verified with vLLM `0.1.dev17053+g7e53283b1`, `torch 2.11.0+cu128`,
> CUDA 12.8, Python 3.12.

## The model

| | |
|---|---|
| Repo | `Qwen/Qwen3.6-35B-A3B-FP8` (served to clients as `Qwen/Qwen3.6-35B-A3B`) |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`) |
| Type | Multimodal (image-text-to-text), used here **text-only** |
| Attention | **Hybrid** — `linear_attention` (Gated DeltaNet) with `full_attention` every 4th layer |
| Quantization | **FP8** (~35 GB on disk; bf16 would be ~70 GB) |
| Active params | ~3B per token (A3B MoE, 256 experts) |

## Prerequisites

- 1× NVIDIA H100 80GB (or comparable 80GB card)
- CUDA toolkit 12.8 at `/usr/local/cuda` (provides `nvcc` for runtime JIT)
- [`uv`](https://docs.astral.sh/uv/)

## 1. Set up the environment

```bash
./setup_env.sh
```

This creates a Python 3.12 venv and installs vLLM from this checkout using the
precompiled CUDA wheel (`VLLM_USE_PRECOMPILED=1`), so no local C++/CUDA build is
needed for Python-only changes. See [`setup_env.sh`](./setup_env.sh).

## 2. Launch the server

```bash
# IMPORTANT: activate the venv — this puts .venv/bin (which contains `ninja`)
# on PATH, which FlashInfer/GDN need to JIT-compile kernels on first run.
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda          # so the JIT can find nvcc

vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

### What the flags do

| Flag | Why |
|---|---|
| `--served-model-name` | Cosmetic alias only — clients use the clean name; the FP8 weights are still what's loaded (`/v1/models` `root` field reveals the real repo). |
| `--language-model-only` | Model is multimodal; this rejects image/video inputs (fine for a text/voice backend) and skips multimodal memory profiling, leaving more KV headroom. The vision encoder weights still load. |
| `--max-model-len 32768` | Right-sized for short voice turns. |
| `--gpu-memory-utilization 0.90` | Yields **32.83 GiB KV cache → ~1.4M tokens, 42.9× concurrency** at 32K context. |
| `--enable-prefix-caching` | Warm conversation prefixes → low TTFT. With this hybrid model, vLLM auto-enables Mamba cache `align` mode (experimental). Default-on in V1. |
| `--enable-chunked-prefill` | Protects TTFT under concurrent load. Default-on in V1. |
| `--reasoning-parser qwen3` | Parses `<think>` blocks when present. |
| `--default-chat-template-kwargs '{"enable_thinking": false}'` | Disables reasoning tokens → fast TTFT (responses come back with `reasoning: null`). |

## 3. Verify

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "In one sentence, what is vLLM?"}],
    "max_tokens": 80
  }' | python -m json.tool
```

A healthy response has `finish_reason: "stop"` and `reasoning: null`.

## Gotchas we hit

1. **`FileNotFoundError: 'ninja'` at startup.** FlashInfer (top-k/top-p sampler) and
   the GDN linear-attention prefill kernel JIT-compile on first run via `ninja`.
   `ninja` ships in `.venv/bin` as a dependency, but invoking `.venv/bin/vllm`
   **directly** doesn't put it on `PATH`. Fix: `source .venv/bin/activate` before
   `vllm serve` (or otherwise add `.venv/bin` to `PATH`). `nvcc` must also be
   reachable — set `CUDA_HOME=/usr/local/cuda`.
2. **First boot is slow (~12 min).** One-time torch.compile + two FlashInfer/GDN JIT
   compiles + CUDA graph capture. These are cached (`~/.cache/vllm`), so later boots
   are faster — though CUDA graph capture re-runs each boot.
   To skip the GDN JIT entirely for faster cold starts: `--gdn-prefill-backend triton`.
3. **HF cache location.** This box overrides it via `HF_HOME=/workspace/.hf`
   (not the default `~/.cache/huggingface`). The 35 GB weights download once and
   are reused across boots (weight load ~12 s from cache).
4. **Untuned FP8 MoE config.** vLLM logs `Using default MoE config. Performance might
   be sub-optimal!` — there's no pre-tuned kernel config for this model's expert
   geometry (256 experts, N=512) on H100. Correctness is fine; throughput is tunable
   later via `benchmark_moe.py`.

---

> This README replaces the upstream vLLM project README for this fork. The original
> is preserved in git history (`git show HEAD~1:README.md`). Upstream docs live at
> <https://docs.vllm.ai>.
