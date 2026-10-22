"""Run one model over one split and record predictions, latency, memory and drop stats.

Examples:
  python run.py --model spacy  --split dev
  python run.py --model gliner --split dev --gliner-threshold 0.4
  python run.py --model ollama --split dev --llm-model qwen3:4b
  python run.py --model ollama --split test --llm-model qwen3:4b      # final, once
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from src import models
from src.common import (PeakMemorySampler, hardware_info, latency_summary, load_jsonl,
                        postprocess, write_jsonl)
from src.scoring import score

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["spacy", "gliner", "ollama"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--warmup", type=int, default=3,
                    help="snippets from dev.jsonl run first and excluded from timing")
    ap.add_argument("--spacy-model")
    ap.add_argument("--gliner-model")
    ap.add_argument("--gliner-threshold", type=float)
    ap.add_argument("--llm-model")
    ap.add_argument("--no-schema", action="store_true",
                    help="ollama: free-text JSON instead of constrained structured output")
    ap.add_argument("--keep-leading-the", action="store_true",
                    help="do not strip a leading 'the' from predicted entities")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rows = load_jsonl(data_dir / f"{args.split}.jsonl")
    warm_rows = load_jsonl(data_dir / "dev.jsonl")[: args.warmup]

    adapter = models.build(args.model, spacy_model=args.spacy_model,
                           gliner_model=args.gliner_model,
                           gliner_threshold=args.gliner_threshold,
                           llm_model=args.llm_model, no_schema=args.no_schema)

    with PeakMemorySampler(extra_process_names=adapter.memory_extra_processes) as mem:
        t0 = time.perf_counter()
        adapter.load()
        load_s = time.perf_counter() - t0
        for r in warm_rows:                       # warm-up: never timed, never scored
            adapter.predict(r["text"])

        preds, raws, latencies = [], [], []
        drops = {}
        for r in rows:
            t = time.perf_counter()
            raw = adapter.predict(r["text"])
            latencies.append(time.perf_counter() - t)
            ents, stats = postprocess(raw, r["text"], strip_article=not args.keep_leading_the)
            for k, v in stats.items():
                drops[k] = drops.get(k, 0) + v
            preds.append({"id": r["id"], "entities": ents})
            raws.append({"id": r["id"], "raw_pairs": raw,
                         "raw_output": getattr(adapter, "last_raw", None)})

    ollama_mem = adapter.loaded_model_memory() if hasattr(adapter, "loaded_model_memory") else None
    # Reported memory: for Ollama, process RSS misses GPU/Metal memory, so add the size
    # Ollama reports for the loaded model (weights + KV cache) to the client's RSS.
    peak_mb = mem.peak_mb
    if ollama_mem:
        client_rss = PeakMemorySampler.process_rss_mb()
        peak_mb = max(mem.peak_mb, client_rss + ollama_mem["size_mb"])

    tag = f"{adapter.name}_{args.split}"
    write_jsonl(preds, out_dir / f"predictions_{tag}.jsonl")
    write_jsonl(raws, out_dir / f"raw_{tag}.jsonl")

    scores = score(rows, preds)
    write_jsonl(scores["errors"], out_dir / f"errors_{tag}.jsonl")
    run = {
        "model_key": args.model,
        "name": adapter.name,
        "split": args.split,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "details": adapter.details(),
        "hardware": hardware_info(),
        "load_s": load_s,
        "warmup_snippets": len(warm_rows),
        "latency": latency_summary(latencies),
        "peak_rss_mb": round(mem.peak_mb, 1),
        "peak_memory_mb": round(peak_mb, 1),
        "ollama_loaded_model": ollama_mem,
        "memory_scope": ("max(RSS of this process + ollama processes, "
                         "this process RSS + model size from ollama /api/ps)"
                         if adapter.memory_extra_processes else "peak RSS of this process"),
        "postprocess": {"strip_leading_the": not args.keep_leading_the, "drops": drops},
        "llm_parse_failures": getattr(adapter, "parse_failures", None),
        "local_score": {k: scores[k] for k in ("overall", "per_label")},
    }
    (out_dir / f"run_{tag}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    if args.split == "test":                      # honest record of how often test was touched
        with open(out_dir / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{run['timestamp']}\t{adapter.name}\t{json.dumps(run['details'], default=str)}\n")

    o, lat = scores["overall"], run["latency"]
    print(f"{adapter.name} on {args.split}: P={o['precision']:.3f} R={o['recall']:.3f} "
          f"F1={o['f1']:.3f} | median {lat['median_s']*1000:.0f} ms, p95 {lat['p95_s']*1000:.0f} ms "
          f"| peak {run['peak_memory_mb']:.0f} MB | drops {drops}"
          + (f" | parse failures {run['llm_parse_failures']}" if args.model == "ollama" else ""))
    print(f"Official score: python score.py {data_dir / (args.split + '.jsonl')} "
          f"{out_dir / ('predictions_' + tag + '.jsonl')} --errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
