from risk_iou.nms import nms
from risk_iou.spans import Span


def test_nms_empty():
    assert nms([]) == []


def test_nms_suppresses_high_overlap():
    spans = [
        Span(0, 4, "rising interest rate risk", score=0.9),
        Span(1, 4, "interest rate risk", score=0.7),   # IoU with first = 3/4 > 0.5
        Span(10, 14, "supply chain disruption", score=0.8),
    ]
    kept = nms(spans, iou_threshold=0.5)
    texts = {s.text for s in kept}
    assert "rising interest rate risk" in texts
    assert "supply chain disruption" in texts
    assert "interest rate risk" not in texts  # suppressed by stronger overlap


def test_nms_keeps_low_overlap():
    spans = [
        Span(0, 3, score=0.9),
        Span(2, 5, score=0.8),  # IoU = 1/4 = 0.25 < 0.5 -> kept
    ]
    kept = nms(spans, iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_keeps_highest_score():
    spans = [
        Span(0, 4, score=0.3),
        Span(0, 4, score=0.95),
        Span(0, 4, score=0.5),
    ]
    kept = nms(spans, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.95


def test_nms_length_bonus_prefers_longer_phrase():
    # "artificial" (len1, p=0.80) nested in "artificial intelligence" (len2, p=0.78)
    frag = Span(0, 1, "artificial", score=0.80)
    phrase = Span(0, 2, "artificial intelligence", score=0.78)
    # without bonus: the fragment wins on raw score
    kept_plain = nms([frag, phrase], iou_threshold=0.5)
    assert kept_plain[0].text == "artificial"
    # with a small length bonus: the longer phrase wins the near tie
    kept_bonus = nms([frag, phrase], iou_threshold=0.5, length_bonus=0.05)
    assert kept_bonus[0].text == "artificial intelligence"


def test_nms_is_deterministic_on_ties():
    spans = [Span(0, 4, score=0.5), Span(5, 9, score=0.5), Span(0, 4, score=0.5)]
    a = nms(spans, 0.5)
    b = nms(spans, 0.5)
    assert [(s.start, s.end) for s in a] == [(s.start, s.end) for s in b]
