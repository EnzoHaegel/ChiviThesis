from risk_iou.evaluation import match, _prf
from risk_iou.spans import Span


def test_match_perfect():
    gt = [Span(0, 3), Span(10, 13)]
    det = [Span(0, 3, score=0.9), Span(10, 13, score=0.8)]
    tp, fp, fn = match(det, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (2, 0, 0)


def test_match_false_positive_and_negative():
    gt = [Span(0, 3), Span(10, 13)]
    det = [Span(0, 3, score=0.9), Span(50, 53, score=0.8)]  # 2nd is wrong
    tp, fp, fn = match(det, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 1)


def test_match_below_iou_threshold_is_fp():
    gt = [Span(0, 10)]
    det = [Span(0, 3, score=0.9)]  # IoU = 3/10 = 0.3 < 0.5
    tp, fp, fn = match(det, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_one_gt_one_detection_each():
    # two detections overlapping the same GT: only one can be TP
    gt = [Span(0, 4)]
    det = [Span(0, 4, score=0.9), Span(0, 4, score=0.5)]
    tp, fp, fn = match(det, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_prf_math():
    m = _prf(tp=8, fp=2, fn=2)
    assert m["precision"] == 0.8
    assert m["recall"] == 0.8
    assert m["f1"] == 0.8


def test_prf_zero_safe():
    m = _prf(0, 0, 0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
