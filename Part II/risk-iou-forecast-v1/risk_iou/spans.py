"""Token spans and 1-D Intersection-over-Union.

A `Span` is the text analogue of an object-detection bounding box: instead of a
pixel rectangle it is a half-open interval ``[start, end)`` over token indices.
IoU between two spans is the standard interval IoU. This module also produces:

  * ground-truth risk spans (exact lexicon matches, greedy longest-first), and
  * candidate detection spans (anchored n-gram region proposals).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import LexEntry
from .text import STOPWORDS, Token, ngrams


@dataclass
class Span:
    start: int                 # token index, inclusive
    end: int                   # token index, exclusive  (half-open [start, end))
    text: str = ""             # surface text (from original document)
    category: str | None = None
    score: float = 1.0

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


def iou(a: Span | tuple[int, int], b: Span | tuple[int, int]) -> float:
    """1-D IoU between two half-open token intervals.

    IoU = |A ∩ B| / |A ∪ B|. Returns 0.0 for disjoint spans, 1.0 for identical
    spans. This is exactly the object-detection IoU specialised to one dimension.
    """
    a0, a1 = (a.start, a.end) if isinstance(a, Span) else a
    b0, b1 = (b.start, b.end) if isinstance(b, Span) else b
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _surface(tokens: list[Token], start: int, end: int) -> str:
    """Reconstruct the original-document substring covered by tokens[start:end)."""
    if start >= end or not tokens:
        return ""
    return _join_surface(tokens, start, end)


def _join_surface(tokens: list[Token], start: int, end: int) -> str:
    return " ".join(t.text for t in tokens[start:end])


def ground_truth_spans(tokens: list[Token], entries: list[LexEntry]) -> list[Span]:
    """Greedy longest-first exact lexicon matching → labeled risk spans.

    Once a token range is claimed by a (longer) phrase, shorter phrases cannot
    re-label the same tokens, so e.g. "rising interest rates" wins over the bare
    "interest rate". This yields the non-overlapping ground-truth label set.
    """
    norms = [t.norm for t in tokens]
    n = len(norms)
    claimed = [False] * n
    spans: list[Span] = []
    # entries are pre-sorted longest-first by lexicon.build_entries()
    for e in entries:
        L = len(e.tokens)
        if L == 0 or L > n:
            continue
        target = e.tokens
        for i in range(n - L + 1):
            if claimed[i] or claimed[i + L - 1]:
                continue
            if tuple(norms[i : i + L]) == target:
                if any(claimed[i : i + L]):
                    continue
                for k in range(i, i + L):
                    claimed[k] = True
                spans.append(
                    Span(
                        start=i,
                        end=i + L,
                        text=_surface(tokens, i, i + L),
                        category=e.category,
                        score=1.0,
                    )
                )
    spans.sort(key=lambda s: s.start)
    return spans


def candidate_spans(
    tokens: list[Token],
    anchors: set[str],
    max_phrase_len: int,
) -> list[Span]:
    """Anchored n-gram region proposals.

    For every n-gram length 1..max_phrase_len we propose the window iff it
    overlaps at least one risk-anchor token. This focuses proposals around
    salient vocabulary (the textual analogue of proposing boxes around salient
    image regions) and keeps the candidate count tractable on long filings.
    """
    norms = [t.norm for t in tokens]
    anchor_pos = [i for i, w in enumerate(norms) if w in anchors]
    if not anchor_pos:
        return []
    anchor_set = set(anchor_pos)
    candidates: list[Span] = []
    seen: set[tuple[int, int]] = set()
    for L in range(1, max_phrase_len + 1):
        for i, j, _ in ngrams(tokens, L):
            # cheap overlap test: does [i, j) contain an anchor?
            if not _window_has_anchor(i, j, anchor_set):
                continue
            i2, j2 = _trim_stopwords(norms, i, j)
            if i2 >= j2:
                continue
            # the trimmed window must still cover an anchor
            if not _window_has_anchor(i2, j2, anchor_set):
                continue
            key = (i2, j2)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Span(start=i2, end=j2, text=_surface(tokens, i2, j2)))
    return candidates


def _trim_stopwords(norms: list[str], start: int, end: int) -> tuple[int, int]:
    """Strip leading/trailing function words from a [start, end) token window."""
    while start < end and norms[start] in STOPWORDS:
        start += 1
    while end > start and norms[end - 1] in STOPWORDS:
        end -= 1
    return start, end


def _window_has_anchor(i: int, j: int, anchor_set: set[int]) -> bool:
    # j - i is small (<= max_phrase_len), so a direct scan is fastest.
    for k in range(i, j):
        if k in anchor_set:
            return True
    return False
