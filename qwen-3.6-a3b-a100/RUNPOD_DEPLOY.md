# RunPod Deployment Guide — qwen-3.6-a3b-a100

Production-ready vLLM server for `Qwen3.6-35B-A3B-Quark-W8A8-INT8` on a single
A100 80 GB, with all our fixes baked in:

- 3 vLLM 0.22 patches auto-applied at build/setup time
- Tuned MoE kernel config bundled (`E=256,N=512,device_name=NVIDIA_PG509-210.json`)
- Extended CUDA graph capture (sizes 1..512 + 640..2048 stride 128) for prefill
- OpenAI-compatible API with streaming, health, metrics, optional API key

**Target TTFT (single-stream, with our fixes applied):**

| L (input tokens) | p50 TTFT (ms) | vs vLLM 0.22 default |
|---:|---:|---:|
|   50 |   55 | −62 % |
|  500 |   56 | −61 % |
| 1000 |   62 | −57 % |
| 2000 |   96 | −33 % |
| 4000 |  166 | −15 % |

See `../runs/FINAL_REPORT.md` for full methodology + comparison plots.

---

## Folder contents

```
qwen-3.6-a3b-a100/
├── Dockerfile             ← build the container
├── setup.sh               ← bare-metal installer (also called by container entrypoint)
├── apply_vllm_patches.sh  ← applies the 3 vLLM patches + nvrtc symlink + bundled MoE config
├── env.sh                 ← runtime LD_LIBRARY_PATH, VLLM_*, PYTORCH_*, sourced everywhere
├── start_server.sh        ← `vllm serve` with our flags (extended CG capture etc.)
├── warmup.py              ← sends N synthetic requests until model is hot
├── client.py              ← minimal CLI client (one prompt / file / stdin)
├── benchmark.py           ← full benchmarker: TTFT p50/p90/p99 + GPU utilization
├── autotune_moe.py        ← Triton MoE kernel autotune (runs auto if no device config)
├── sample_prompts.jsonl   ← example input format for benchmark.py
├── config/                ← bundled tuned MoE config for A100 PG509-210
└── RUNPOD_DEPLOY.md       ← this file
```

Everything needed to deploy is **in this one folder.** Model weights and the
Python venv are installed to configurable external paths (`/data/models` and
`/app/venv` by default).

---

## Path A — Docker (recommended on RunPod)

### 1. Build the image (~10 min, ~7 GB)

```bash
cd qwen-3.6-a3b-a100/
docker build -t qwen-a100:latest .
```

If you push to a registry (DockerHub / ghcr / RunPod private registry), tag and
push so RunPod can pull it directly:
```bash
docker tag qwen-a100:latest <your-registry>/qwen-a100:latest
docker push <your-registry>/qwen-a100:latest
```

### 2. Create a RunPod persistent volume

In the RunPod UI:
- Storage → New Network Volume
- Size: **40 GB minimum** (32 GB for model + headroom)
- Region: same as your GPU pod
- Name e.g. `qwen-models`

This keeps the 33 GB model on disk across pod restarts (instead of re-downloading every cold start).

### 3. Launch a Pod

- Template: **Custom Image** → your pushed image
- GPU: **1× A100 80 GB**
- Container Disk: 20 GB (for image + venv overflow)
- Volume mount: `/data/models` → `qwen-models` (from step 2)
- Expose port: **8000** (HTTP)
- Env vars:
  - `HF_TOKEN` = `hf_xxx…` (recommended for faster, reliable model download)
  - `API_KEY` = `sk-xxx…` (optional; require auth on the API)

### 4. First-start behavior

On a fresh volume the container will:
1. Detect `/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8/` is empty
2. Download model from HuggingFace (~5–10 min, faster with `HF_TOKEN`)
3. Auto-fix the Quark tokenizer bug (swap `tokenizer_config.json`)
4. Check if a device-specific MoE config exists for this GPU. For PG509-210 (A100) one is bundled — skipped. For ANY OTHER GPU it'll auto-autotune (~30 min, one-time).
5. Launch `vllm serve` — another ~3 min for CUDA-graph capture
6. Server ready at `http://<pod-ip>:8000`

Subsequent starts (volume already populated): skip 1-4, just step 5 → ~3 min ready.

### 5. Verify

```bash
# health
curl http://<pod-ip>:8000/health

# list models
curl http://<pod-ip>:8000/v1/models

# one-shot completion
curl http://<pod-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d '{
    "model": "qwen-3.6-a3b",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

---

## Path B — Bare-metal RunPod pod (no Docker)

If you launched a stock PyTorch RunPod template and want to install manually:

```bash
cd /workspace
git clone <your-repo>          # or scp this folder over
cd qwen-3.6-a3b-a100/

# Optional: pin the HF token + model path
export HF_TOKEN=hf_xxx
export MODEL_DIR=/workspace/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8
export VENV_DIR=/workspace/venv

bash setup.sh                  # one-shot: deps + patches + model + tokenizer fix
source env.sh                  # LD_LIBRARY_PATH etc.
bash start_server.sh           # serves at :8000
```

Total bare-metal first-run: ~15-25 min depending on network speed for the model.

---

## Using it — Client

```bash
# one-shot completion
python client.py "What is 2 + 2?"

# streaming chat
python client.py --chat --stream "Explain RAG in two sentences."

# batch from file
python client.py --input my_prompts.jsonl

# auth
python client.py --api-key sk-... --base-url http://<pod-ip>:8000/v1 "Hello"
```

---

## Using it — Benchmark

The benchmark script takes an input file and reports TTFT (p50/p90/p99), total
latency, GPU utilization, etc. It is single-stream (one request at a time)
matching the voice-agent scenario.

**Input file format** (auto-detected by extension):

- `*.jsonl` — one JSON per line:
  ```json
  {"prompt": "What is your name?"}
  {"prompt": "Translate to Spanish: hello", "max_tokens": 30}
  {"messages": [{"role": "user", "content": "..."}], "system": "Voice agent."}
  {"prompt_token_ids": [123, 456, 789]}   ← reproducible by exact token IDs
  ```
- `*.json` — JSON list of the above objects, OR list of strings.
- `*.txt` — one prompt per line.

**Run**:

```bash
# voice-agent-style: TTFT only, streaming, multiple trials per prompt
python benchmark.py \
  --input sample_prompts.jsonl \
  --out bench_voice \
  --stream --max-tokens 1 \
  --warmup 5 --trials 5

# full generation (TTFT + TPOT)
python benchmark.py \
  --input sample_prompts.jsonl \
  --out bench_full \
  --stream --chat --max-tokens 256 \
  --warmup 3 --trials 1
```

**Outputs** (under `--out` dir):
- `per_call.csv` — every request's TTFT/total/in-out tokens
- `gpu_samples.csv` — nvidia-smi samples at 200ms intervals during the bench
- `summary.json` — aggregated p50/p90/p99 + GPU util summary
- Console table with headline numbers

Example `summary.json`:
```json
{
  "n_requests": 30,
  "ttft_ms": { "min": 55.0, "p50": 95.4, "p90": 167.3, "p99": 186.1, "max": 198.2 },
  "tpot_ms_p50": 5.12,
  "gpu": { "gpu_util_avg": 76.4, "gpu_util_max": 98.0, "mem_used_max_mib": 68512 }
}
```

---

## Configuration knobs

Set via env var, all optional:

| Var | Default | What it does |
|---|---|---|
| `MODEL_DIR` | `/data/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8` | Where model weights live (persist across runs) |
| `PORT` | `8000` | HTTP port |
| `API_KEY` | (unset) | If set, requires `Authorization: Bearer <key>` |
| `MAX_MODEL_LEN` | `6300` | Max prompt+gen tokens. Increase if you handle longer prompts (more KV memory) |
| `MAX_NUM_SEQS` | `4` | Concurrent in-flight requests. ↑ if you want batch throughput, ↓ if memory tight |
| `GPU_MEM_UTIL` | `0.85` | Fraction of GPU for vLLM. Lower to leave room for other procs |
| `EXTENDED_CG` | `1` | `0` to disable the extended prefill captures (saves ~3 min cold start, costs perf at L=512-2048) |
| `HF_TOKEN` | (unset) | HuggingFace token; recommended for download speed |
| `SERVED_MODEL_NAME` | `qwen-3.6-a3b` | Name clients use in `model=...` field |
| `SKIP_MODEL` | `0` | Skip model download in `setup.sh` (assume already at MODEL_DIR) |
| `SKIP_AUTOTUNE` | `0` | Skip MoE autotune in `setup.sh` (use vLLM heuristic, slower) |

---

## Observability

- `GET /health` — 200 once vLLM has loaded the model and is accepting requests. Use as RunPod healthcheck.
- `GET /metrics` — Prometheus format. Includes per-request latencies, queue depth, KV cache utilization, etc. Scrape every 15s if you want to graph.
- `GET /v1/models` — confirm the served model name.

Container `STDOUT` has all vLLM logs (including init phases, kernel selections, error traces).

---

## Cost / perf cheat-sheet (single A100 80 GB on RunPod)

Assumption: $1.50/hr A100 spot.

| Workload | TTFT (p50) | Tokens/sec/request | Cost per 1M tokens |
|---|---:|---:|---:|
| Voice agent, ≤500 token prompts, 1-tok response | **56 ms** | n/a (streaming) | dominated by request count, not tokens |
| Voice agent, 2K prompts, 1-tok response | **96 ms** | n/a | same |
| RAG, 4K prompts, 256-tok response | 166 ms TTFT + ~50 tok/s gen | ~50 | ~$0.83 |

---

## Troubleshooting

### "Model not found" on startup
`SKIP_MODEL=0` (default) should auto-download. If it doesn't:
- Check pod has internet (in RunPod sometimes a region has limited egress)
- Set `HF_TOKEN` even for public models — anonymous DLs get rate-limited
- Manually run inside container: `bash /app/setup.sh`

### Cold start very slow
First start at a new pod or new GPU type:
1. Model download (5-10 min)
2. CUDA graph capture (~3 min for default + extended sizes)
3. Autotune on first non-A100 GPU (~30 min, ONE-TIME — cached to volume)

After first start: ~3 min.

### "OutOfMemory" on startup
You hit the cudagraph workspace cliff (see `FINAL_REPORT.md` section 4). On
A100 80 GB with INT8 W8A8 model, captures ≤ 2048 fit but 3072+ doesn't.

- Don't override `EXTENDED_CG=1` with `cudagraph_capture_sizes` that go ≥ 3072
- If you need 3072+ captures: switch to PP=2 across 2 GPUs (set `--tensor-parallel-size 1 --pipeline-parallel-size 2` in start_server.sh)

### TTFT higher than the table above
- First 3–5 requests are warm-up — bench excludes them. Use `--warmup 5`.
- Confirm tuned MoE config loaded: look for `Using configuration from .../E=256,N=512,...json` in logs.
- Confirm extended CG loaded: look for `cudagraph_capture_sizes: ... max=2048` in startup log.
- Confirm INT8 path active: look for `Selected CutlassInt8ScaledMMLinearKernel for QuarkW8A8Int8`.

### API requests hang / time out
- Check `curl http://localhost:8000/health` — server may still be loading
- Increase RunPod healthcheck `start-period` to 600s
- Check `/metrics` for queue depth — if growing, your client is firing faster than the server processes

---

## Porting to H100 / different GPU

The code is GPU-agnostic. Only one device-specific artifact ships in `config/`
(the A100 tuned MoE JSON). When you start the container on a non-A100 GPU,
`setup.sh` will:
1. Detect the GPU name
2. Look for `E=256,N=512,device_name=<GPU>.json` in vLLM's config dir
3. If not found AND `SKIP_AUTOTUNE != 1` → run `autotune_moe.py` (~30 min, one-time)

After first start, the new device config persists to the venv (or, if you bake
it into the persistent volume, across pod restarts). Set `SKIP_AUTOTUNE=1` for a
quick (suboptimal) first start if you can't wait.

**Bonus on H100:** native FP8 tensor cores ~1979 TFLOPS. Consider also trying
the Qwen FP8 variant with vLLM's cutlass FP8 MoE backend — could outperform our
INT8 W8A8 path on H100. See `FINAL_REPORT.md` §7 for details.

---

## Production checklist

- [ ] Persistent volume mounted for `/data/models` (otherwise 33 GB redownload per cold start)
- [ ] `HF_TOKEN` set (for download reliability)
- [ ] `API_KEY` set (otherwise the API is open)
- [ ] RunPod healthcheck pointed at `/health` with start-period ≥ 600s
- [ ] Logs being captured by RunPod / shipped to your aggregator
- [ ] Metrics scraper pointed at `/metrics` if you want graphs
- [ ] Test request from outside the pod confirms the API works
- [ ] Run `benchmark.py` once to confirm TTFT matches table
- [ ] For voice agent: client uses **streaming** and reads the first chunk fast

---

## What's NOT in this kit (open follow-ups)

See `runs/FINAL_REPORT.md` §10 for the full open-followups list:
- Quality eval of the W8A8 community quant vs Qwen BF16 base (MMLU, GSM8K, etc.) — flagged as critical before production use
- Debugging the 3072+ cudagraph memory cliff — would unlock the predicted ~140 ms TTFT @ L=4K
- Filing the 3 vLLM PRs upstream (drafts in `../upstream_pr/`)
- PP=2 mode for 2× A100 deployments
