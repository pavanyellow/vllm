"""Generate a TTFT sweep report from benchmark.py output.

Reads per_call.csv (from benchmark.py), groups by n_in (input token count),
computes p50/p90/p99 TTFT per length, writes summary CSV + Plotly HTML report.

Usage:
  python make_sweep_report.py --input sweep_results/per_call.csv --out sweep_report
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

A100_HBM_BW_EFF = 2.0 * 0.85 * 1e12
A100_BF16_FLOPS_EFF = 312 * 0.80 * 1e12
MODEL_WEIGHT_GB = 33.5
ACTIVE_PARAMS = 3.0e9

MEM_FLOOR_MS = (MODEL_WEIGHT_GB * 1e9) / A100_HBM_BW_EFF * 1000
COMPUTE_US_PER_TOK = (2 * ACTIVE_PARAMS / A100_BF16_FLOPS_EFF) * 1e6

SERIES_COLORS = [
    ("rgb(31,119,180)", "rgba(31,119,180,0.15)"),
    ("rgb(255,127,14)", "rgba(255,127,14,0.15)"),
]


def load_per_call(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "n_in": int(r["n_in"]),
                "ttft_ms": float(r["ttft_ms"]),
            })
    return rows


def summarize(rows):
    by_length = {}
    for r in rows:
        by_length.setdefault(r["n_in"], []).append(r["ttft_ms"])

    summary = []
    for length in sorted(by_length):
        vals = sorted(by_length[length])
        n = len(vals)
        def q(p):
            return vals[min(int(p * n), n - 1)]
        summary.append({
            "length": length,
            "n": n,
            "min_ms": vals[0],
            "p50_ms": q(0.50),
            "p90_ms": q(0.90),
            "p99_ms": q(0.99),
            "mean_ms": statistics.mean(vals),
            "std_ms": statistics.stdev(vals) if n >= 2 else 0.0,
        })
    return summary


def write_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


def build_html(summary, title="TTFT Sweep — Qwen3.6-35B-A3B W8A8 INT8 on A100"):
    lengths = [s["length"] for s in summary]
    p50 = [s["p50_ms"] for s in summary]
    p90 = [s["p90_ms"] for s in summary]
    mins = [s["min_ms"] for s in summary]
    roofline = [max(MEM_FLOOR_MS, COMPUTE_US_PER_TOK * L / 1000) for L in lengths]

    n_total = sum(s["n"] for s in summary)

    target_lengths = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    table_rows = ""
    for s in summary:
        if s["length"] in target_lengths:
            table_rows += (
                f"<tr><td>{s['length']}</td><td>{s['p50_ms']:.1f}</td>"
                f"<td>{s['p90_ms']:.1f}</td><td>{s['p99_ms']:.1f}</td>"
                f"<td>{s['min_ms']:.1f}</td><td>{s['n']}</td></tr>"
            )

    full_table = ""
    for s in summary:
        full_table += (
            f"<tr><td>{s['length']}</td><td>{s['n']}</td>"
            f"<td>{s['min_ms']:.1f}</td><td>{s['p50_ms']:.1f}</td>"
            f"<td>{s['p90_ms']:.1f}</td><td>{s['p99_ms']:.1f}</td>"
            f"<td>{s['mean_ms']:.1f}</td><td>{s['std_ms']:.1f}</td></tr>"
        )

    # KPI values
    kpi_500 = next((s["p50_ms"] for s in summary if s["length"] == 500), 0)
    kpi_2000 = next((s["p50_ms"] for s in summary if s["length"] == 2000), 0)
    kpi_4000 = next((s["p50_ms"] for s in summary if s["length"] == 4000), 0)
    kpi_6000 = next((s["p50_ms"] for s in summary if s["length"] == 6000), 0)

    plot_data = [
        {"x": lengths, "y": mins, "mode": "lines", "line": {"width": 0},
         "showlegend": False, "hoverinfo": "skip", "name": "min"},
        {"x": lengths, "y": p90, "mode": "lines", "line": {"width": 0},
         "fill": "tonexty", "fillcolor": "rgba(31,119,180,0.18)",
         "name": "min–p90 band", "hoverinfo": "skip"},
        {"x": lengths, "y": p50, "mode": "lines+markers",
         "line": {"color": "rgb(31,119,180)", "width": 2.5},
         "marker": {"size": 5}, "name": "measured p50",
         "hovertemplate": "L=%{x}<br>p50=%{y:.1f} ms<extra></extra>"},
        {"x": lengths, "y": roofline, "mode": "lines",
         "line": {"color": "rgb(214,39,40)", "width": 1.8, "dash": "dash"},
         "name": "A100 theoretical floor",
         "hovertemplate": "L=%{x}<br>floor=%{y:.1f} ms<extra></extra>"},
    ]
    layout = {
        "xaxis": {"title": "prompt length (tokens, log scale)", "type": "log",
                  "tickvals": [1, 10, 100, 1000, 10000],
                  "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        "yaxis": {"title": "TTFT (ms)", "rangemode": "tozero",
                  "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        "hovermode": "x unified", "template": "plotly_white",
        "height": 540,
        "margin": {"l": 70, "r": 30, "t": 20, "b": 60},
        "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.85)"},
    }

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --fg: #1a1a1a; --muted: #666; --border: #e4e4e4; --bg-soft: #f8f8f8; --blue: #1f77b4; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0; color: var(--fg); background: #fff; line-height: 1.55; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 36px 28px 60px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 36px 0 10px; padding-top: 14px; border-top: 1px solid var(--border); }}
  p {{ margin: 8px 0; }}
  code {{ background: var(--bg-soft); padding: 1px 6px; border-radius: 3px; font-size: 0.92em;
          font-family: "SF Mono", Menlo, Consolas, monospace; }}
  .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0 32px; }}
  .kpi {{ background: var(--bg-soft); padding: 18px 14px; border-radius: 8px;
          border: 1px solid var(--border); text-align: center; }}
  .kpi-val {{ font-size: 24px; font-weight: 600; color: var(--blue); }}
  .kpi-lbl {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }}
  th, td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: right; }}
  th {{ background: var(--bg-soft); font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  details {{ margin: 16px 0; border: 1px solid var(--border); border-radius: 6px;
             padding: 10px 14px; background: var(--bg-soft); }}
  details summary {{ cursor: pointer; font-weight: 500; color: var(--muted); }}
  details[open] {{ background: #fff; }}
  .formula {{ background: var(--bg-soft); padding: 10px 14px; border-left: 3px solid var(--blue);
             font-family: "SF Mono", Menlo, monospace; font-size: 13px; margin: 8px 0; }}
</style>
</head><body>
<div class="wrap">

<h1>{title}</h1>
<div class="meta">
  GPU: <code>NVIDIA A100 80GB SXM</code> ·
  Model: <code>Qwen3.6-35B-A3B-Quark-W8A8-INT8</code> ·
  vLLM 0.22 + patches + MoE autotune + extended CUDA graphs ·
  {n_total} measurements across {len(summary)} lengths ·
  <code>max_tokens=1</code>, streaming, greedy
</div>

<div class="kpis">
  <div class="kpi"><div class="kpi-val">{kpi_500:.0f} ms</div>
    <div class="kpi-lbl">TTFT p50<br>L=500</div></div>
  <div class="kpi"><div class="kpi-val">{kpi_2000:.0f} ms</div>
    <div class="kpi-lbl">TTFT p50<br>L=2000</div></div>
  <div class="kpi"><div class="kpi-val">{kpi_4000:.0f} ms</div>
    <div class="kpi-lbl">TTFT p50<br>L=4000</div></div>
  <div class="kpi"><div class="kpi-val">{kpi_6000:.0f} ms</div>
    <div class="kpi-lbl">TTFT p50<br>L=6000</div></div>
</div>

<h2>TTFT scaling curve</h2>
<div id="plot"></div>
<p style="font-size: 13px; color: var(--muted)">
  Red dashed line = A100 theoretical floor: <code>max(mem_floor {MEM_FLOOR_MS:.1f}ms, compute {COMPUTE_US_PER_TOK:.1f}µs/tok × L)</code>.
  Shaded band = min to p90. Hover for exact values.
</p>

<h2>Key lengths</h2>
<table>
  <thead><tr><th>length</th><th>p50 ms</th><th>p90 ms</th><th>p99 ms</th><th>min ms</th><th>n</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>

<h2>Roofline model</h2>
<div class="formula">
  Memory floor = {MODEL_WEIGHT_GB} GB / (0.85 × 2.0 TB/s) = {MEM_FLOOR_MS:.1f} ms<br>
  Compute slope = 2 × 3B / (0.80 × 312 TFLOPS) = {COMPUTE_US_PER_TOK:.1f} µs/token<br>
  Ridge point ≈ {int(MEM_FLOOR_MS * 1000 / COMPUTE_US_PER_TOK)} tokens
</div>

<h2>Full data</h2>
<details>
  <summary>All {len(summary)} lengths</summary>
  <table>
    <thead><tr><th>length</th><th>n</th><th>min</th><th>p50</th><th>p90</th><th>p99</th><th>mean</th><th>std</th></tr></thead>
    <tbody>{full_table}</tbody>
  </table>
</details>

</div>
<script>
  Plotly.newPlot('plot', {json.dumps(plot_data)}, {json.dumps(layout)},
                 {{responsive: true, displaylogo: false}});
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="per_call.csv from benchmark.py")
    p.add_argument("--out", default="sweep_report", help="output directory")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_per_call(args.input)
    print(f"loaded {len(rows)} measurements")

    summary = summarize(rows)
    print(f"grouped into {len(summary)} lengths")

    csv_path = out / "summary.csv"
    write_summary_csv(summary, csv_path)
    print(f"wrote {csv_path}")

    html_path = out / "ttft_sweep_report.html"
    html = build_html(summary)
    html_path.write_text(html)
    print(f"wrote {html_path} ({len(html) / 1024:.1f} KB)")

    print("\n=== HIGHLIGHTS ===")
    for s in summary:
        if s["length"] in [1, 50, 500, 1000, 2000, 4000, 6000]:
            print(f"  L={s['length']:>5}  p50={s['p50_ms']:>7.1f}ms  p90={s['p90_ms']:>7.1f}ms  p99={s['p99_ms']:>7.1f}ms")


if __name__ == "__main__":
    main()
