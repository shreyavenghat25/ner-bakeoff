# Local NER Bake-off

Compares three locally-run NER models on the provided synthetic dataset: quality, latency, memory and estimated cost.

| Key | Model | Family | Why it is in the comparison |
|---|---|---|---|
| `spacy` | `en_core_web_sm` 3.8.0 (spaCy 3.8.16) | Specialist, fixed labels | Cheapest possible option: tiny, CPU-only, fast. Fixed OntoNotes labels must be mapped to ours. |
| `gliner` | `urchade/gliner_medium-v2.1`, 195M params, fp32 (gliner 0.2.29, torch 2.14.1) | Specialist, zero-shot | Takes our label names as input, so no label-mapping gap. Still CPU-friendly. |
| `ollama` | `qwen3:4b`, 4.0B params, Q4_K_M GGUF (Ollama model digest `359d7dd4bcda`) | Local LLM | Follows written annotation rules from a prompt. The "can a small LLM do it?" option. |

## Results (test set, 50 snippets, 108 gold entities)

F1 values below are from the official `score.py` (micro-averaged, exact match). Each model was run on test exactly once (`results/test_runs.log`).

| Model | P | R | F1 | F1 PERSON | F1 ORG | F1 LOCATION | F1 EVENT | Median latency (s) | p95 latency (s) | Peak memory (MB) | Est. cost / 1k snippets (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spaCy `en_core_web_sm` | 0.670 | 0.620 | 0.644 | 0.727 | 0.621 | 0.658 | 0.421 | 0.003 | 0.004 | 308 | 0.0001 |
| GLiNER medium v2.1 (threshold 0.5) | 0.740 | 0.870 | 0.800 | 0.772 | 0.780 | 0.943 | 0.500 | 0.060 | 0.064 | 1,518 | 0.0030 |
| Qwen3-4B Q4_K_M (Ollama) | 0.971 | 0.935 | 0.953 | 0.941 | 0.980 | 0.963 | 0.903 | 1.966 | 3.475 | 3,048 | 0.46 |

Cost assumptions (US East on-demand Linux, checked 2026-10-20, see `costs.json` for sources): spaCy on EC2 c7i.large ($0.0892/h), GLiNER on c7i.xlarge ($0.1785/h), Qwen3-4B on g6.xlarge with one NVIDIA L4 ($0.8048/h). Cost uses the mean latency measured on the laptop, one snippet at a time. For the LLM this applies M2 latency to an L4 price, so it is a rough, probably pessimistic, cross-hardware estimate. It leaves out idle time, batching and concurrency, engineering and maintenance effort, storage and data transfer.

Full table with assumptions: `results/results_test.md` / `.csv`. Discussion and recommendation: `NOTE.md`.

## Hardware

MacBook Air, Apple M2 (8 cores), 8 GB unified memory, macOS 14 (Darwin 23.4.0), Python 3.11.17 (conda). spaCy and GLiNER ran on CPU. Ollama ran Qwen3-4B on the M2's integrated GPU via Metal (the `GPU: none detected` line in `results_test.md` only checks for NVIDIA GPUs).

## Setup

Tested with Python 3.11.17. Use a virtual environment (results here were produced in a conda environment).

```bash
conda create -n ner python=3.11 -y && conda activate ner
# or: python3.11 -m venv .venv && source .venv/bin/activate
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
python run.py --model gliner --split test --gliner-threshold 0.5
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

## Assumptions and dev-set decisions

- Leading "the" is stripped, based on the brief's example ("Gardens by the Bay"). This is wrong for names that genuinely start with "The". Checked on dev with spaCy: F1 0.875 with stripping vs 0.792 without (a difference of two entities on a 24-entity dev set), so stripping was kept for all models.
- spaCy label mapping: GPE, LOC, FAC → LOCATION. NORP (demonyms), WORK_OF_ART, DATE and others are dropped because the brief says not to tag them.
- GLiNER label prompts: `person`, `organization`, `location`, `named event`. Threshold: 0.5 (the library default). On dev, 0.5 and 0.6 tied (F1 0.885) and 0.4 scored 0.868, a one-entity difference, which is noise on 10 snippets, so the default was kept.

## Reproducibility notes

- `requirements.txt` pins library versions. The run JSON records the exact versions used.
- The LLM uses temperature 0 and seed 42. Ollama can still vary slightly across versions and hardware.
- Hardware: see the Hardware section above. `run_*.json` records it per run.
- Latency excludes model load and 3 warm-up snippets from dev. Memory for Ollama combines client RSS with the loaded-model size reported by Ollama (3,028 MB for Qwen3-4B, weights + 4,096-token KV cache).

## AI-assistant disclosure

The evaluation harness code (model adapters, post-processing, timing and memory measurement, results summary), the cloud price lookup and this README were drafted with an AI assistant (Claude). I set up and ran all experiments on my own laptop, made the dev-set decisions, reviewed the code, and wrote the analysis in `NOTE.md`.
