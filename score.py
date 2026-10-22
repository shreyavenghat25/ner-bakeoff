#!/usr/bin/env python3
"""Entity-level NER scorer for the NER bake-off.

Usage:
    python score.py data/test.jsonl predictions.jsonl [--errors]

Prediction format: one JSON object per line
    {"id": "t01", "entities": [{"text": "Daniel Koh", "label": "PERSON"}]}

Matching rules (do not modify when reporting results):
  * An entity is the pair (text, label); text is compared case-insensitively
    after trimming whitespace. Labels must match exactly.
  * Duplicates within one snippet count once.
  * Missing ids are scored as "no entities predicted" and reported.
  * Entities with labels outside PERSON/ORG/LOCATION/EVENT are counted as
    false positives.
"""
import argparse, json, sys
from collections import defaultdict

LABELS = ["PERSON", "ORG", "LOCATION", "EVENT"]

def norm(e):
    return (str(e.get("text", "")).strip().lower(), str(e.get("label", "")).strip().upper())

def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows[r["id"]] = {norm(e) for e in r.get("entities", [])}
            except Exception as ex:
                print(f"warning: {path} line {ln} unreadable ({ex})", file=sys.stderr)
    return rows

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold"); ap.add_argument("pred")
    ap.add_argument("--errors", action="store_true", help="list false positives / negatives")
    a = ap.parse_args()

    gold, pred = load(a.gold), load(a.pred)
    missing = [i for i in gold if i not in pred]
    extra = [i for i in pred if i not in gold]
    c = defaultdict(lambda: [0, 0, 0])  # label -> tp, fp, fn
    errs = []
    for i, g in gold.items():
        p = pred.get(i, set())
        for e in g & p: c[e[1]][0] += 1; c["ALL"][0] += 1
        for e in p - g: c[e[1]][1] += 1; c["ALL"][1] += 1; errs.append((i, "FP", e))
        for e in g - p: c[e[1]][2] += 1; c["ALL"][2] += 1; errs.append((i, "FN", e))

    print(f"snippets scored: {len(gold)}  missing predictions: {len(missing)}  unknown ids: {len(extra)}")
    print(f"{'label':<10}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>6}{'FN':>6}")
    for k in LABELS + ["ALL"]:
        tp, fp, fn = c[k]
        p, r, f = prf(tp, fp, fn)
        name = "MICRO-ALL" if k == "ALL" else k
        print(f"{name:<10}{p:>7.3f}{r:>7.3f}{f:>7.3f}{tp:>6}{fp:>6}{fn:>6}")
    other = [k for k in c if k not in LABELS + ["ALL"]]
    if other:
        print("note: predictions used labels outside the set:", ", ".join(other))
    if a.errors:
        print("\nerrors (id, type, entity):")
        for i, t, e in errs:
            print(f"  {i}  {t}  {e[0]!r} [{e[1]}]")

if __name__ == "__main__":
    main()
