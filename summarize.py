"""Collect results/run_*_<split>.json into one results table (Markdown + CSV).

Cost per 1,000 snippets = hourly price x (1,000 x mean seconds per snippet) / 3,600.
It uses MEAN latency (throughput), measured one snippet at a time on the local machine.
See the README for what this leaves out.

  python summarize.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ("PERSON", "ORG", "LOCATION", "EVENT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--costs", default=str(ROOT / "costs.json"))
    args = ap.parse_args()

    costs = json.loads(Path(args.costs).read_text(encoding="utf-8"))
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(args.results_dir).glob(f"run_*_{args.split}.json"))]
    if not runs:
        raise SystemExit(f"No run_*_{args.split}.json files in {args.results_dir}")

    header = (["Model", "Size / quantisation", "F1"] + [f"F1 {l}" for l in LABELS] +
              ["Median latency (s)", "p95 latency (s)", "Peak memory (MB)",
               "Cost / 1k snippets (USD)", "Cost assumption"])
    rows = []
    for r in runs:
        d, s, lat = r["details"], r["local_score"], r["latency"]
        if r["model_key"] == "ollama":
            size = f"{d.get('parameter_size')}, {d.get('quantization')}"
        elif "params_millions" in d:
            size = f"{d['params_millions']}M params, {d.get('precision')}"
        else:
            size = f"{d.get('model')} v{d.get('model_version')}, {d.get('precision')}"
        c = costs.get(r["model_key"], {})
        price = c.get("usd_per_hour")
        if price:
            cost = f"{price * 1000 * lat['mean_s'] / 3600:.4f}"
            assumption = f"{c.get('instance')} @ ${price}/h ({c.get('source') or 'source missing!'})"
        else:
            cost, assumption = "n/a", "set usd_per_hour + source in costs.json"
        rows.append([r["name"], size, f"{s['overall']['f1']:.3f}"] +
                    [f"{s['per_label'][l]['f1']:.3f}" for l in LABELS] +
                    [f"{lat['median_s']:.3f}", f"{lat['p95_s']:.3f}", f"{r['peak_memory_mb']:.0f}",
                     cost, assumption])

    out = Path(args.results_dir)
    with open(out / f"results_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    hw = runs[0]["hardware"]
    md += ["", f"Hardware: {hw['cpu']}, {hw['cpu_logical_cores']} logical cores, {hw['ram_gb']} GB RAM, "
               f"GPU: {hw['gpu'] or 'none detected'}, {hw['os']}, Python {hw['python']}.",
           f"F1 here is from the local scorer; confirm every number with the official score.py."]
    (out / f"results_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
