# Note: Local NER Bake-off

> Template. Rename to NOTE.md and replace every prompt in italics with your own words, from your own runs. About one page. The guiding questions are there to help; delete them before submitting.

## Results

_Paste `results/results_test.md` here, with F1 numbers taken from the official `score.py`._

## Recommendation

**Customer who cares about cost:** _Which model? Use your cost-per-1k and F1 numbers. Is the cheapest model good enough, and on which labels does it fall short?_

**Customer who cares mainly about quality:** _Which model? How big is the quality gap, and is it bigger than the noise you would expect with 50 snippets?_

_Optional: is there a hybrid, e.g. a cheap model first and an LLM only for some cases?_

## One failure pattern

_Pick ONE pattern from `results/errors_*_test.jsonl`. Name it, say which model(s) it affects and roughly how often, and show 2–3 snippets with gold vs predicted. Say why you think it happens (label-set mismatch? span boundaries? prompt rules not followed?)._

_State that you looked at test errors to write this section, as the brief asks._

## Is the metric fair?

_Exact case-insensitive matching. Did you see predictions that were arguably correct but scored as wrong (e.g. "the X" vs "X", or a defensible label choice)? Give an example if so. Don't change score.py; just describe it._

## What I would test next with another day

_Two or three concrete experiments, each with the question it answers. For example: another LLM size or quantisation, few-shot vs zero-shot prompt, GLiNER threshold sweep, batching for throughput._

## Limitations: how much should you trust these numbers?

_Be specific. Things to consider: 50 test snippets (one snippet can move per-label F1 a lot, especially EVENT); synthetic data vs real customer text; one run on one laptop; latency at batch size 1 on CPU vs a cloud GPU; how the cost estimate was built and what it leaves out (idle time, batching, concurrency, engineering effort, storage, data transfer); post-processing choices that help some models more than others; how many times test was run (`results/test_runs.log`)._

## AI-assistant disclosure

_One sentence on where you used AI assistants._
