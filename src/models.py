"""Model adapters. Each returns raw (text, label) pairs; cleaning is done in common.postprocess.

Three families, as the brief requires:
  spacy  - specialist, fixed OntoNotes label set mapped to ours (fast CPU baseline)
  gliner - specialist, zero-shot: takes our label names as input (runs on CPU)
  ollama - local LLM via the Ollama HTTP API, prompted for JSON
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class Adapter:
    name = "base"
    memory_extra_processes: tuple[str, ...] = ()

    def load(self) -> None: ...
    def predict(self, text: str) -> list[tuple[str, str]]: raise NotImplementedError
    def details(self) -> dict: return {}


# ------------------------------------------------------------------ spaCy
class SpacyAdapter(Adapter):
    """OntoNotes labels -> ours. NORP (demonyms), DATE, CARDINAL, WORK_OF_ART etc. are
    dropped on purpose because the brief says not to tag them. FAC (buildings, venues,
    landmarks) and LOC (rivers, mountains) both count as LOCATION in our scheme."""

    LABEL_MAP = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION",
                 "FAC": "LOCATION", "EVENT": "EVENT"}

    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model
        self.name = f"spacy-{model}"

    def load(self):
        import spacy
        self.nlp = spacy.load(self.model_name, disable=["lemmatizer"])

    def predict(self, text):
        doc = self.nlp(text)
        return [(e.text, self.LABEL_MAP[e.label_]) for e in doc.ents if e.label_ in self.LABEL_MAP]

    def details(self):
        import spacy
        return {"library": f"spacy {spacy.__version__}", "model": self.model_name,
                "model_version": self.nlp.meta.get("version"), "precision": "fp32 (CPU)",
                "label_map": self.LABEL_MAP}


# ------------------------------------------------------------------ GLiNER
class GlinerAdapter(Adapter):
    """Zero-shot NER. The label *descriptions* are part of the model input, so they are a
    tunable setting (tune on dev only), as is the confidence threshold."""

    DEFAULT_LABELS = {
        "person": "PERSON",
        "organization": "ORG",
        "location": "LOCATION",
        "named event": "EVENT",
    }

    def __init__(self, model: str = "urchade/gliner_medium-v2.1", threshold: float = 0.5,
                 labels: dict[str, str] | None = None):
        self.model_name = model
        self.threshold = threshold
        self.labels = labels or self.DEFAULT_LABELS
        self.name = "gliner-" + model.split("/")[-1]

    def load(self):
        import torch
        from gliner import GLiNER
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model = GLiNER.from_pretrained(self.model_name)
        self.model.eval()

    def predict(self, text):
        ents = self.model.predict_entities(text, list(self.labels), threshold=self.threshold)
        return [(e["text"], self.labels[e["label"]]) for e in ents]

    def details(self):
        import gliner
        import torch
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"library": f"gliner {getattr(gliner, '__version__', '?')}, torch {torch.__version__}",
                "model": self.model_name, "params_millions": round(n_params / 1e6),
                "precision": "fp32 (CPU)", "threshold": self.threshold,
                "label_prompts": self.labels}


# ------------------------------------------------------------------ Ollama LLM
_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": ["PERSON", "ORG", "LOCATION", "EVENT"]},
                },
                "required": ["text", "label"],
            },
        }
    },
    "required": ["entities"],
}


def parse_llm_json(raw: str) -> tuple[list[tuple[str, str]], bool]:
    """Tolerant parser for LLM output. Returns (pairs, parsed_ok).

    Handles: <think> blocks, ```json fences, prose around the JSON, a bare list instead of
    {"entities": [...]}, and single objects. Anything unparseable becomes an empty
    prediction and is counted, never crashes the run.
    """
    s = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    candidates = [s]
    m = re.search(r"\{.*\}|\[.*\]", s, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            items = obj.get("entities", [obj] if "text" in obj else [])
        elif isinstance(obj, list):
            items = obj
        else:
            continue
        pairs = [(str(i.get("text", "")), str(i.get("label", "")))
                 for i in items if isinstance(i, dict)]
        return pairs, True
    return [], False


class OllamaAdapter(Adapter):
    memory_extra_processes = ("ollama",)

    def __init__(self, model: str = "qwen3:4b", host: str = "http://localhost:11434",
                 prompt_file: str = "ner_prompt.txt", use_schema: bool = True):
        self.model_name = model
        self.host = host.rstrip("/")
        self.prompt = (PROMPT_DIR / prompt_file).read_text(encoding="utf-8")
        self.prompt_file = prompt_file
        self.use_schema = use_schema
        self.name = "ollama-" + model.replace(":", "-").replace("/", "-")
        self.parse_failures = 0
        self.last_raw = ""

    def load(self):
        r = requests.post(f"{self.host}/api/show", json={"model": self.model_name}, timeout=30)
        r.raise_for_status()
        self._show = r.json()
        # Force the model into memory so load time is not counted as latency.
        requests.post(f"{self.host}/api/generate",
                      json={"model": self.model_name, "prompt": "", "keep_alive": "30m"},
                      timeout=600).raise_for_status()

    def predict(self, text):
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self.prompt.replace("{text}", text)}],
            "stream": False,
            "think": False,                 # Qwen3 etc.: no reasoning trace, faster + parseable
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }
        if self.use_schema:
            body["format"] = _SCHEMA        # Ollama structured outputs (constrained decoding)
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=300)
        if r.status_code == 400 and "think" in r.text:   # models without a thinking switch
            body.pop("think")
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        self.last_raw = r.json().get("message", {}).get("content", "")
        pairs, ok = parse_llm_json(self.last_raw)
        if not ok:
            self.parse_failures += 1
        return pairs

    def loaded_model_memory(self) -> dict | None:
        """Ask Ollama how much memory the loaded model occupies (GET /api/ps).

        Needed because on Apple Silicon (Metal) and NVIDIA GPUs the weights live in GPU
        memory, which does not appear in the server process's RSS.
        """
        try:
            r = requests.get(f"{self.host}/api/ps", timeout=10)
            r.raise_for_status()
            for m in r.json().get("models", []):
                if m.get("name") == self.model_name or m.get("model") == self.model_name:
                    return {"size_mb": round(m.get("size", 0) / 2**20, 1),
                            "size_vram_mb": round(m.get("size_vram", 0) / 2**20, 1),
                            "context_length": m.get("context_length")}
        except Exception:
            pass
        return None

    def details(self):
        d = self._show.get("details", {})
        return {"runtime": "ollama", "model": self.model_name,
                "parameter_size": d.get("parameter_size"),
                "quantization": d.get("quantization_level"),
                "family": d.get("family"), "format": d.get("format"),
                "prompt_file": self.prompt_file, "structured_output": self.use_schema,
                "temperature": 0, "seed": 42}


def build(model: str, **kw) -> Adapter:
    if model == "spacy":
        return SpacyAdapter(kw.get("spacy_model") or "en_core_web_sm")
    if model == "gliner":
        return GlinerAdapter(kw.get("gliner_model") or "urchade/gliner_medium-v2.1",
                             threshold=kw.get("gliner_threshold") or 0.5)
    if model == "ollama":
        return OllamaAdapter(kw.get("llm_model") or "qwen3:4b",
                             use_schema=not kw.get("no_schema", False))
    raise ValueError(f"unknown model {model!r}")
