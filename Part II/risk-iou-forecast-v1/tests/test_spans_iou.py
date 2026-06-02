from risk_iou.spans import Span, iou, ground_truth_spans, candidate_spans
from risk_iou.lexicon import build_entries, anchor_tokens
from risk_iou.text import tokenize


def test_iou_identical():
    assert iou(Span(0, 5), Span(0, 5)) == 1.0


def test_iou_disjoint():
    assert iou(Span(0, 3), Span(5, 8)) == 0.0


def test_iou_half_overlap():
    # [0,4) and [2,6): inter=2, union=6 -> 1/3
    assert abs(iou(Span(0, 4), Span(2, 6)) - (2 / 6)) < 1e-9


def test_iou_touching_is_zero():
    # half-open intervals that touch at a point do not overlap
    assert iou(Span(0, 3), Span(3, 6)) == 0.0


def test_iou_tuple_and_span_interchangeable():
    assert iou((0, 4), Span(2, 6)) == iou(Span(0, 4), (2, 6))


def test_ground_truth_greedy_longest_first():
    entries = build_entries()
    text = "Our rising interest rates exposure and interest rate risk are material."
    toks = tokenize(text)
    gt = ground_truth_spans(toks, entries)
    phrases = {g.text.lower() for g in gt}
    # the long phrase should win and claim its tokens
    assert any("rising interest rates" in p for p in phrases)
    # spans must not overlap each other
    spans = sorted((g.start, g.end) for g in gt)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0


def test_candidate_spans_are_anchored():
    entries = build_entries()
    anchors = anchor_tokens(entries)
    text = "The quick brown fox jumped over inflation and supply chain disruption."
    toks = tokenize(text)
    cands = candidate_spans(toks, anchors, max_phrase_len=4)
    # every candidate must contain at least one anchor token
    norms = [t.norm for t in toks]
    for c in cands:
        assert any(norms[i] in anchors for i in range(c.start, c.end))
    # and we should find at least the obvious risk windows
    assert any(c.text.lower() == "inflation" for c in cands)
