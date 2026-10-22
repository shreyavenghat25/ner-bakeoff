"""Local re-implementation of the brief's metric, used for quick feedback and the results table.

The provided score.py is the official number. This mirrors its documented rule (exact
match on case-insensitive text + label, repeated mentions count once). If the two ever
disagree, report score.py and investigate the difference.
"""
from __future__ import annotations

from collections import defaultdict

from .common import LABELS


def _keyset(ents):
    return {(e["text"].strip().lower(), e["label"]) for e in ents}


def score(gold_rows: list[dict], pred_rows: list[dict]) -> dict:
    preds = {r["id"]: r.get("entities", []) for r in pred_rows}
    missing = [g["id"] for g in gold_rows if g["id"] not in preds]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    errors = []
    for g in gold_rows:
        gs, ps = _keyset(g["entities"]), _keyset(preds.get(g["id"], []))
        for _, lab in gs & ps:
            tp[lab] += 1
        for _, lab in ps - gs:
            fp[lab] += 1
        for _, lab in gs - ps:
            fn[lab] += 1
        if gs != ps:
            errors.append({"id": g["id"], "text": g["text"],
                           "false_positives": sorted(ps - gs), "false_negatives": sorted(gs - ps)})

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f, "tp": t, "fp": f_p, "fn": f_n}

    per_label = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in LABELS}
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))   # micro-average
    return {"overall": overall, "per_label": per_label, "missing_ids": missing, "errors": errors}
