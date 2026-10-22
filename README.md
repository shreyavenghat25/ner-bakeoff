# Local NER Bake-off

Compares three locally-run NER models on the provided synthetic dataset: quality, latency, memory and estimated cost.

| Key | Model | Family | Why it is in the comparison |
|---|---|---|---|
| `spacy` | `en_core_web_sm` 3.8.0 | Specialist, fixed labels | Cheapest possible option: tiny, CPU-only, fast. Fixed OntoNotes labels must be mapped to ours. |
| `gliner` | `urchade/gliner_medium-v2.1` | Specialist, zero-shot | Takes our label names as input, so no label-mapping gap. Still CPU-friendly. |
| `ollama` | `qwen3:4b` (Q4_K_M) via Ollama | Local LLM | Follows written annotation rules from a prompt. The "can a small LLM do it?" option. |

> Fill in the exact versions you ran from `results/run_*.json` (it records library versions, model size and quantisation automatically).

## Setup

Tested with Python 3.11. Use a virtual environment.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the LLM, install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen3:4b
ollama serve            # if it is not already running in the background
```

Put the provided files in place:

```
data/dev.jsonl
data/test.jsonl
score.py
```

Check the output-handling code before running anything real:

```bash
python -m tests.test_postprocess
```

## How to run

**1. Develop on dev only.** Change one setting at a time and note each result in a dev log.

```bash
python run.py --model spacy  --split dev
python run.py --model gliner --split dev --gliner-threshold 0.5
python run.py --model ollama --split dev --llm-model qwen3:4b
python run.py --model spacy  --split dev --keep-leading-the    # check the "the" assumption
```

**2. Run each final configuration on test once.** Every test run is appended to `results/test_runs.log`, so the number of times test was touched is on record.

```bash
python run.py --model spacy  --split test
python run.py --model gliner --split test --gliner-threshold <chosen on dev>
python run.py --model ollama --split test --llm-model qwen3:4b
```

**3. Score with the official script** (these are the numbers to report):

```bash
python score.py data/test.jsonl results/predictions_<name>_test.jsonl --errors
```

**4. Cost.** `costs.json` holds the hourly price assumptions with sources (US East on-demand, checked 2026-10-20; confirm on the AWS pricing page). Build the table:

```bash
python summarize.py --split test      # writes results/results_test.md and .csv
```

## What each run writes (`results/`)

| File | Contents |
|---|---|
| `predictions_<name>_<split>.jsonl` | Predictions in the brief's format, ready for `score.py` |
| `run_<name>_<split>.json` | Model details, hardware, load time, latency (median, p95, mean), peak memory, post-processing drops, LLM parse failures, local scores |
| `errors_<name>_<split>.jsonl` | Per-snippet false positives and false negatives for failure analysis |
| `raw_<name>_<split>.jsonl` | Raw model output before cleaning (for the LLM, the exact text it returned) |
| `test_runs.log` | One line per test-split run |

## Method

**Same post-processing for every model** (`src/common.py`), so differences come from the models, not from per-model cleanup:

- Possessives stripped (`Singapore's` → `Singapore`).
- Honorifics removed from PERSON (`Minister Daniel Koh` → `Daniel Koh`).
- A leading "the" removed (assumption below).
- Entities whose text is not in the snippet are dropped (guards against LLM hallucination) and counted.
- Labels outside the four allowed are dropped and counted. Common aliases (`ORGANIZATION`, `LOC`) are mapped.
- Duplicates removed (repeated mentions count once).
- "Longest name only": an entity that appears *only* inside a longer predicted entity is dropped. If it also appears on its own elsewhere in the snippet, it is kept.

**LLM output handling** (`src/models.py`): temperature 0, fixed seed, thinking disabled, and Ollama structured outputs (a JSON schema constrains decoding). The parser still tolerates code fences, prose around the JSON, a bare list, and `<think>` blocks. Unparseable output is scored as "no entities" and counted as a parse failure, not hidden. `--no-schema` turns off constrained decoding, which is useful as a dev experiment.

**Latency**: wall-clock per snippet, one snippet at a time (batch size 1). Model load and a 3-snippet warm-up (taken from dev, never test) are excluded.

**Memory**: peak RSS sampled every 50 ms. For Ollama, RSS misses the model weights on Apple Silicon (Metal) and NVIDIA GPUs, so the reported figure also uses the loaded-model size that Ollama itself reports (`/api/ps`, weights + KV cache). Both numbers are stored in `run_*.json` (`peak_rss_mb`, `ollama_loaded_model`).

**Cost per 1,000 snippets** = hourly price × (1,000 × mean seconds per snippet) / 3,600. Mean is used, not median, because cost follows total time.

## Assumptions (edit to reflect your own decisions)

- Leading "the" is stripped, based on the brief's example ("Gardens by the Bay"). This is wrong for names that genuinely start with "The". Its effect was checked on dev: _fill in_.
- spaCy label mapping: GPE, LOC, FAC → LOCATION. NORP (demonyms), WORK_OF_ART, DATE and others are dropped because the brief says not to tag them.
- GLiNER label prompts: `person`, `organization`, `location`, `named event`. Threshold chosen on dev: _fill in_.

## Reproducibility notes

- `requirements.txt` pins library versions. The run JSON records the exact versions used.
- The LLM uses temperature 0 and seed 42. Ollama can still vary slightly across versions and hardware.
- Hardware used: _fill in from `run_*.json`_.

## AI-assistant disclosure

_Fill in one honest sentence, e.g. which parts of the code were drafted with an AI assistant, and confirm that the experiments, results and note are your own._
