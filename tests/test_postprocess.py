"""Unit tests for the output-handling code. Run: python -m tests.test_postprocess"""
from src.common import percentile, postprocess
from src.models import parse_llm_json
from src.scoring import score


def keys(ents):
    return {(e["text"], e["label"]) for e in ents}


def test_possessive_and_honorific():
    text = "Minister Daniel Koh praised Singapore's port."
    ents, _ = postprocess([("Minister Daniel Koh", "PERSON"), ("Singapore's", "LOCATION")], text)
    assert keys(ents) == {("Daniel Koh", "PERSON"), ("Singapore", "LOCATION")}


def test_longest_span_only():
    text = "She attended the Busan International Film Festival."
    ents, st = postprocess([("Busan International Film Festival", "EVENT"), ("Busan", "LOCATION")], text)
    assert keys(ents) == {("Busan International Film Festival", "EVENT")} and st["nested"] == 1


def test_inner_name_kept_when_also_standalone():
    text = "Singapore Airlines flew from Singapore."
    ents, _ = postprocess([("Singapore Airlines", "ORG"), ("Singapore", "LOCATION")], text)
    assert keys(ents) == {("Singapore Airlines", "ORG"), ("Singapore", "LOCATION")}


def test_hallucination_bad_label_and_duplicates():
    text = "Priya Nair visited Tuas."
    ents, st = postprocess([("Priya Nair", "PERSON"), ("priya nair", "PERSON"),
                            ("Jurong", "LOCATION"), ("Tuas", "PLACE")], text)
    assert keys(ents) == {("Priya Nair", "PERSON")}
    assert st == {"bad_label": 1, "not_in_text": 1, "empty": 0, "duplicate": 1, "nested": 0}


def test_leading_article():
    text = "Roads near the Esplanade were closed by the Land Transport Authority."
    raw = [("the Esplanade", "LOCATION"), ("the Land Transport Authority", "ORG")]
    on, _ = postprocess(raw, text)
    off, _ = postprocess(raw, text, strip_article=False)
    assert keys(on) == {("Esplanade", "LOCATION"), ("Land Transport Authority", "ORG")}
    assert keys(off) == {("the Esplanade", "LOCATION"), ("the Land Transport Authority", "ORG")}


def test_label_aliases():
    ents, _ = postprocess([("Acme", "organization"), ("Oslo", "LOC")], "Acme opened in Oslo.")
    assert keys(ents) == {("Acme", "ORG"), ("Oslo", "LOCATION")}


def test_llm_parser_messy_outputs():
    ok = [
        '{"entities": [{"text": "Oslo", "label": "LOCATION"}]}',
        '```json\n{"entities": [{"text": "Oslo", "label": "LOCATION"}]}\n```',
        '<think>hmm</think>Sure! Here you go: {"entities": [{"text": "Oslo", "label": "LOCATION"}]} Hope it helps',
        '[{"text": "Oslo", "label": "LOCATION"}]',
    ]
    for raw in ok:
        pairs, parsed = parse_llm_json(raw)
        assert parsed and pairs == [("Oslo", "LOCATION")], raw
    assert parse_llm_json('{"entities": []}') == ([], True)
    assert parse_llm_json("I could not find any entities.") == ([], False)
    assert parse_llm_json('{"entities": [{"text": "Oslo", "label": ') == ([], False)


def test_scorer():
    gold = [{"id": "a", "text": "x", "entities": [{"text": "Oslo", "label": "LOCATION"},
                                                  {"text": "Acme", "label": "ORG"}]},
            {"id": "b", "text": "y", "entities": []}]
    pred = [{"id": "a", "entities": [{"text": "oslo", "label": "LOCATION"},
                                     {"text": "Acme", "label": "PERSON"}]},
            {"id": "b", "entities": [{"text": "Zed", "label": "PERSON"}]}]
    s = score(gold, pred)["overall"]
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1)
    assert abs(s["precision"] - 1 / 3) < 1e-9 and s["recall"] == 0.5


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert abs(percentile(list(range(1, 21)), 95) - 19.05) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
