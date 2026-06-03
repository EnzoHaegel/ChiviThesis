from risk_iou.gold import (
    phrases_to_spans,
    parse_gold_annotations,
    _prf,
)
import pandas as pd


def test_phrases_to_spans_basic():
    sent = "Rising interest rates and inflation could harm our business."
    spans = phrases_to_spans(sent, ["interest rates", "inflation"])
    texts = {s.text.lower() for s in spans}
    assert "interest rates" in texts
    assert "inflation" in texts
    # offsets are contiguous token subsequences
    for s in spans:
        assert s.end > s.start


def test_phrases_to_spans_case_insensitive():
    sent = "COVID-19 disrupted our SUPPLY chain."
    spans = phrases_to_spans(sent, ["covid-19", "supply chain"])
    assert len(spans) == 2


def test_phrases_to_spans_missing_phrase_ignored():
    sent = "Interest rate risk is material."
    spans = phrases_to_spans(sent, ["cyber attack"])  # not present
    assert spans == []


def test_phrases_to_spans_multiple_occurrences():
    sent = "tariffs and more tariffs hurt margins."
    spans = phrases_to_spans(sent, ["tariffs"])
    assert len(spans) == 2


def test_phrases_to_spans_empty_and_blank():
    sent = "Some text here."
    assert phrases_to_spans(sent, ["", "  "]) == []


def test_parse_gold_annotations():
    df = pd.DataFrame(
        {
            "sent_uid": [0, 1],
            "gold_risk_terms": ["inflation; tariffs", ""],
        }
    )
    out = parse_gold_annotations(df)
    assert out[0] == ["inflation", "tariffs"]
    assert out[1] == []


def test_prf_helper():
    m = _prf(3, 1, 1)
    assert m["precision"] == 0.75
    assert m["recall"] == 0.75
    assert m["f1"] == 0.75
