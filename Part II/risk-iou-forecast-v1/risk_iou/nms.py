"""Non-Maximum Suppression over 1-D token spans.

Direct transposition of object-detection NMS: sort detections by score, greedily
keep the highest-scoring span, and suppress any remaining span that overlaps a
kept span too much. This removes duplicated / overlapping keyword detections
(e.g. "interest rate", "rising interest rate", "rising interest rate risk"
collapsing to the single strongest detection).

Text keyword spans differ from image boxes in one important way: candidates are
heavily *nested* (a 1-token window inside a 3-token window). Plain IoU does not
suppress nesting strongly enough (IoU of [0,1) inside [0,3) is only 1/3), so the
same risk would be reported several times. We therefore suppress on BOTH:
  * IoU > iou_threshold                          (object-detection criterion), and
  * containment = |A∩B| / min(|A|,|B|) >= containment_threshold  (nesting).
"""
from __future__ import annotations

from .spans import Span, iou


def _containment(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter == 0:
        return 0.0
    return inter / min(a.length, b.length)


def nms(
    spans: list[Span],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    length_bonus: float = 0.0,
) -> list[Span]:
    """Return the kept spans after greedy NMS.

    A weaker span is suppressed by a kept span when their IoU exceeds
    ``iou_threshold`` OR when one is largely contained in the other
    (``containment_threshold``). Set ``containment_threshold`` to 1.01 to recover
    pure IoU NMS.

    ``length_bonus`` adds ``length_bonus * (len - 1)`` to a span's score *for
    ordering only* (the reported ``span.score`` is untouched). With a small
    value this makes the more specific / longer phrase win over a contained
    fragment when their model scores are close — e.g. keeping
    "artificial intelligence" rather than the bare "artificial" — which yields
    cleaner extracted keywords. Default 0.0 = pure score ordering.

    Stable w.r.t. ties (earlier start, then shorter span) so the result is
    deterministic.
    """
    if not spans:
        return []

    def eff(i: int) -> float:
        return spans[i].score + length_bonus * (spans[i].length - 1)

    order = sorted(
        range(len(spans)),
        key=lambda i: (-eff(i), spans[i].start, spans[i].length),
    )
    suppressed = [False] * len(spans)
    keep: list[Span] = []
    for pos, idx in enumerate(order):
        if suppressed[idx]:
            continue
        keep.append(spans[idx])
        kept_span = spans[idx]
        for jdx in order[pos + 1 :]:
            if suppressed[jdx]:
                continue
            other = spans[jdx]
            if (
                iou(kept_span, other) > iou_threshold
                or _containment(kept_span, other) >= containment_threshold
            ):
                suppressed[jdx] = True
    keep.sort(key=lambda s: s.start)
    return keep
