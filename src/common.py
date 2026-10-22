"""Shared helpers: data I/O, entity post-processing, timing, memory sampling, hardware info.

Every model adapter returns raw (text, label) pairs. All cleaning happens here, in one
place, so the three models are post-processed identically and the comparison stays fair.
"""
from __future__ import annotations

import json
import platform
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil

LABELS = ("PERSON", "ORG", "LOCATION", "EVENT")


# ---------------------------------------------------------------- data I/O
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- post-processing
_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
_HONORIFICS = (
    "mr", "mrs", "ms", "dr", "prof", "professor", "sir", "dame", "minister",
    "president", "senator", "mayor", "governor", "captain", "coach", "judge",
)
_HONORIFIC_RE = re.compile(r"^(?:%s)\.?\s+" % "|".join(_HONORIFICS), re.IGNORECASE)


def clean_text(text: str, label: str, strip_article: bool = True) -> str:
    """Normalise one entity string according to the brief's annotation rules.

    strip_article removes a leading "the" ("the Land Transport Authority" ->
    "Land Transport Authority"). The brief's gold example tags "Gardens by the Bay", not
    "the Gardens by the Bay", so stripping is the default. It hurts names that really start
    with "The" ("The Straits Times"): check the effect on dev with --keep-leading-the.
    """
    t = text.strip().strip(" ,.;:!?\"“”()[]")
    if strip_article and t.lower().startswith("the ") and len(t) > 4:
        t = t[4:].strip()
    t = _POSSESSIVE.sub("", t).strip()          # "Singapore's" -> "Singapore"
    if label == "PERSON":                       # "Minister Daniel Koh" -> "Daniel Koh"
        t = _HONORIFIC_RE.sub("", t).strip()
    return t


def postprocess(raw: list[tuple[str, str]], source_text: str,
                require_in_text: bool = True,
                strip_article: bool = True) -> tuple[list[dict], dict]:
    """Clean, validate, dedupe and keep longest spans.

    Returns (entities, stats). stats counts what was dropped and why, so messy model
    output is visible in the results instead of silently disappearing.
    """
    stats = {"bad_label": 0, "not_in_text": 0, "empty": 0, "duplicate": 0, "nested": 0}
    lowered_src = source_text.lower()
    seen: dict[tuple[str, str], dict] = {}
    for text, label in raw:
        label = (label or "").strip().upper()
        label = {"ORGANIZATION": "ORG", "ORGANISATION": "ORG", "LOC": "LOCATION",
                 "PER": "PERSON"}.get(label, label)
        if label not in LABELS:
            stats["bad_label"] += 1
            continue
        t = clean_text(str(text), label, strip_article)
        if not t:
            stats["empty"] += 1
            continue
        if require_in_text and t.lower() not in lowered_src:   # hallucination guard
            stats["not_in_text"] += 1
            continue
        key = (t.lower(), label)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen[key] = {"text": t, "label": label}

    # "Tag the longest name only": drop an entity that occurs ONLY inside a longer
    # predicted entity ("Busan" inside "Busan International Film Festival"). If it also
    # appears on its own elsewhere in the snippet ("Singapore Airlines ... from Singapore"),
    # it is a separate mention and is kept.
    ents = list(seen.values())
    kept = []
    for e in ents:
        e_low = e["text"].lower()
        longer = [o["text"].lower() for o in ents
                  if len(o["text"]) > len(e["text"]) and e_low in o["text"].lower()]
        if not longer:
            kept.append(e)
            continue
        remainder = lowered_src
        for o in sorted(longer, key=len, reverse=True):
            remainder = remainder.replace(o, " ")
        inner = not re.search(r"(?<!\w)%s(?!\w)" % re.escape(e_low), remainder)
        if inner:
            stats["nested"] += 1
        else:
            kept.append(e)
    return kept, stats


# ---------------------------------------------------------------- metrics
def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_summary(latencies: list[float]) -> dict:
    return {
        "n": len(latencies),
        "median_s": statistics.median(latencies),
        "p95_s": percentile(latencies, 95),
        "mean_s": statistics.fmean(latencies),
        "total_s": sum(latencies),
    }


class PeakMemorySampler:
    """Samples RSS of this process (+ children, + any extra matching processes) in a thread.

    For Ollama the model lives in a separate server process, so pass
    extra_process_names=("ollama",) to include it. On Apple Silicon this RSS
    approximates unified memory used; for an NVIDIA GPU also check nvidia-smi.
    """

    def __init__(self, interval: float = 0.05, extra_process_names: tuple[str, ...] = ()):
        self.interval = interval
        self.extra = tuple(n.lower() for n in extra_process_names)
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _current(self) -> int:
        me = psutil.Process()
        procs = [me] + me.children(recursive=True)
        if self.extra:
            for p in psutil.process_iter(["name"]):
                name = (p.info.get("name") or "").lower()
                if any(x in name for x in self.extra) and p.pid != me.pid:
                    procs.append(p)
        total = 0
        for p in {p.pid: p for p in procs}.values():
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._current())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_bytes = self._current()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @staticmethod
    def process_rss_mb() -> float:
        return psutil.Process().memory_info().rss / 2**20

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 2**20


def hardware_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "gpu": None,
    }
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"] = out.stdout.strip()
    except Exception:
        pass
    return info
