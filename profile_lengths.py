#!/usr/bin/env python3
"""Capture torch-profiler traces at several cold prefill lengths."""
import glob, os, random, shutil, time
import requests

URL = "http://127.0.0.1:8080"
TRACES = "/workspace/traces"

def ids(n):
    return [random.randrange(100, 150000) for _ in range(n)]

def req(prompt):
    t0 = time.perf_counter()
    with requests.post(f"{URL}/v1/completions", json={
            "model": "glm-5.2-fp8", "prompt": prompt, "max_tokens": 5,
            "temperature": 0.0, "ignore_eos": True, "stream": True},
            stream=True) as r:
        r.raise_for_status()
        for _ in r.iter_lines():
            return (time.perf_counter() - t0) * 1000

def capture(tag, prompt):
    for f in glob.glob(f"{TRACES}/*"):
        if os.path.isfile(f):
            os.remove(f)
    requests.post(f"{URL}/start_profile").raise_for_status()
    ttft = req(prompt)
    requests.post(f"{URL}/stop_profile").raise_for_status()
    time.sleep(20)
    dest = f"{TRACES}/{tag}"
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in glob.glob(f"{TRACES}/*"):
        if os.path.isfile(f):
            shutil.move(f, dest); n += 1
    print(f"{tag}: ttft {ttft:.1f} ms ({n} files)", flush=True)

for _ in range(3):
    req(ids(4000))
for L in (1000, 2000, 4000, 8000):
    capture(f"L{L}", ids(L))
print("done")
