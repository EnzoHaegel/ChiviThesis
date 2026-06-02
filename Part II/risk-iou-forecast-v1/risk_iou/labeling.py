"""Phase 1 — IoU-based detection, extraction and labeling.

For every document we:
  1. tokenise and (optionally) truncate,
  2. build ground-truth risk spans from the lexicon (greedy longest-first),
  3. propose anchored candidate spans (region proposals),
  4. label each candidate by its IoU against the ground-truth spans
     (positive iff max-IoU >= match threshold), attaching the GT category.

The flattened candidate table is the supervised dataset for Phase 2; the
ground-truth table is the "cleaned & labeled risk dataset" deliverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .data import Filing, read_text
from .lexicon import LexEntry, anchor_tokens, build_entries
from .spans import Span, candidate_spans, ground_truth_spans, iou
from .text import Token, tokenize


@dataclass
class DocAnnotation:
    filing: Filing
    tokens: list[Token]
    gt: list[Span]
    candidates: list[Span]
    labels: list[int] = field(default_factory=list)          # 1 = risk, 0 = not
    cand_categories: list[str | None] = field(default_factory=list)
    cand_best_iou: list[float] = field(default_factory=list)


def _truncate(tokens: list[Token], max_tokens: int) -> list[Token]:
    return tokens[:max_tokens] if max_tokens and len(tokens) > max_tokens else tokens


def annotate_document(
    filing: Filing,
    cfg: Config,
    entries: list[LexEntry],
    anchors: set[str],
) -> DocAnnotation:
    text = read_text(filing)
    tokens = _truncate(tokenize(text), cfg.max_tokens_per_doc)

    gt = ground_truth_spans(tokens, entries)
    cands = candidate_spans(tokens, anchors, cfg.max_phrase_len)

    labels: list[int] = []
    cats: list[str | None] = []
    best_ious: list[float] = []
    for c in cands:
        best_iou = 0.0
        best_cat: str | None = None
        for g in gt:
            v = iou(c, g)
            if v > best_iou:
                best_iou = v
                best_cat = g.category
        labels.append(1 if best_iou >= cfg.match_iou_threshold else 0)
        cats.append(best_cat if best_iou >= cfg.match_iou_threshold else None)
        best_ious.append(best_iou)

    return DocAnnotation(
        filing=filing,
        tokens=tokens,
        gt=gt,
        candidates=cands,
        labels=labels,
        cand_categories=cats,
        cand_best_iou=best_ious,
    )


def annotate_corpus(
    filings: list[Filing],
    cfg: Config,
    entries: list[LexEntry] | None = None,
    anchors: set[str] | None = None,
) -> list[DocAnnotation]:
    if entries is None:
        entries = build_entries()
    if anchors is None:
        anchors = anchor_tokens(entries)
    return [annotate_document(f, cfg, entries, anchors) for f in filings]


def candidates_dataframe(annotations: list[DocAnnotation]) -> pd.DataFrame:
    """Flat supervised table: one row per candidate span."""
    rows = []
    for ann in annotations:
        f = ann.filing
        for c, y, cat, biou in zip(
            ann.candidates, ann.labels, ann.cand_categories, ann.cand_best_iou
        ):
            rows.append(
                {
                    "filing_id": f.filing_id,
                    "cik": f.cik,
                    "date": f.date,
                    "quarter": f.quarter,
                    "form": f.form,
                    "span_text": c.text,
                    "token_start": c.start,
                    "token_end": c.end,
                    "span_len": c.length,
                    "best_iou": round(biou, 4),
                    "category": cat,
                    "label": y,
                }
            )
    return pd.DataFrame(rows)


def ground_truth_dataframe(annotations: list[DocAnnotation]) -> pd.DataFrame:
    """The cleaned & labeled risk dataset: one row per ground-truth risk span."""
    rows = []
    for ann in annotations:
        f = ann.filing
        for g in ann.gt:
            rows.append(
                {
                    "filing_id": f.filing_id,
                    "cik": f.cik,
                    "date": f.date,
                    "quarter": f.quarter,
                    "form": f.form,
                    "risk_term": " ".join(t.norm for t in ann.tokens[g.start : g.end]),
                    "surface": g.text,
                    "category": g.category,
                    "token_start": g.start,
                    "token_end": g.end,
                }
            )
    return pd.DataFrame(rows)
