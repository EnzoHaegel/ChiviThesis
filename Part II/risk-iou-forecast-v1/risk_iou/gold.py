"""Gold-standard validation harness (Phase-3 robustness check).

The Phase-3 metrics are computed against *weak-supervision* labels derived from
the risk lexicon. To check that those labels are a reasonable proxy for genuine
risk keywords — and not merely self-consistent with the detector — we build an
independent gold set:

  1. sample sentences stratified by year (`sample_sentences`),
  2. a human (or LLM) annotator marks the genuine risk-keyword *phrases* in each
     sentence — by typing the phrase strings, NOT token offsets,
  3. `phrases_to_spans` maps each annotated phrase back to token spans,
  4. `evaluate_against_gold` scores BOTH the trained detector AND the lexicon
     weak-supervision against the gold, via the same IoU matching as Phase 3.

The phrase-string annotation format is deliberately annotator-friendly and is
independent of the seed lexicon, so lexicon-vs-gold agreement is informative.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .config import Config
from .data import Filing, discover_filings, read_text
from .evaluation import match
from .lexicon import LexEntry, anchor_tokens, build_entries
from .model import RiskDetectorModel
from .nms import nms
from .spans import Span, candidate_spans, ground_truth_spans
from .text import Token, split_sentences, tokenize


@dataclass
class GoldSentence:
    filing_id: str
    year: int
    quarter: str
    sent_idx: int
    text: str


def _year_bucket(year: int) -> str:
    if year <= 2012:
        return "<=2012"
    if year <= 2020:
        return "2013-2020"
    return ">=2021"


def sample_sentences(
    cfg: Config,
    n: int = 90,
    seed: int = 7,
    min_tokens: int = 6,
    max_tokens: int = 60,
    filings: list[Filing] | None = None,
) -> list[GoldSentence]:
    """Sample `n` sentences stratified across year buckets for annotation."""
    if filings is None:
        filings = discover_filings(cfg)
    rng = random.Random(seed)
    rng.shuffle(filings)

    # bucket filings by year band
    buckets: dict[str, list[Filing]] = {}
    for f in filings:
        buckets.setdefault(_year_bucket(f.year), []).append(f)

    per_bucket = max(1, n // max(len(buckets), 1))
    collected: list[GoldSentence] = []
    for band, band_filings in buckets.items():
        got = 0
        for f in band_filings:
            if got >= per_bucket:
                break
            text = read_text(f)
            sents = split_sentences(text)
            # take a couple of usable sentences per filing to diversify firms
            usable = [
                (i, s)
                for i, s in enumerate(sents)
                if min_tokens <= len(tokenize(s)) <= max_tokens
            ]
            if not usable:
                continue
            rng.shuffle(usable)
            for sent_idx, s in usable[:2]:
                if got >= per_bucket:
                    break
                collected.append(
                    GoldSentence(
                        filing_id=f.filing_id,
                        year=f.year,
                        quarter=f.quarter,
                        sent_idx=sent_idx,
                        text=" ".join(s.split()),
                    )
                )
                got += 1
    rng.shuffle(collected)
    return collected[:n]


def worksheet_dataframe(
    sentences: list[GoldSentence], suggestions: bool = True
) -> pd.DataFrame:
    """Annotation worksheet: one row per sentence; annotator fills gold_risk_terms."""
    entries = build_entries()
    rows = []
    for k, gs in enumerate(sentences):
        sug = ""
        if suggestions:
            sug = "; ".join(cue_based_risk_phrases(gs.text, entries))
        rows.append(
            {
                "sent_uid": k,
                "filing_id": gs.filing_id,
                "year": gs.year,
                "quarter": gs.quarter,
                "sentence": gs.text,
                "suggested_risk_terms": sug,
                "gold_risk_terms": "",  # annotator fills: phrases separated by ';'
            }
        )
    return pd.DataFrame(rows)


# --- independent (lexicon-free) cue-based suggester --------------------------
_RISK_CUES = (
    "adversely",
    "could harm",
    "may harm",
    "could affect",
    "negatively affect",
    "risk",
    "risks",
    "uncertainty",
    "uncertainties",
    "exposed to",
    "subject to",
    "could result",
    "materially",
    "disrupt",
    "volatility",
    "decline",
    "failure",
)


def cue_based_risk_phrases(sentence: str, entries: list[LexEntry]) -> list[str]:
    """A weak, lexicon-independent suggester to pre-fill the worksheet.

    It is NOT used for evaluation — only to lower annotation effort. It flags
    sentences carrying risk discourse cues and proposes short noun-ish chunks
    around them. The annotator overrides freely.
    """
    low = sentence.lower()
    if not any(c in low for c in _RISK_CUES):
        return []
    toks = tokenize(sentence)
    norms = [t.norm for t in toks]
    # propose 2-3 grams ending just before a cue word like "risk"/"uncertainty"
    proposals: list[str] = []
    cue_heads = {"risk", "risks", "uncertainty", "uncertainties", "volatility", "disruption"}
    for i, w in enumerate(norms):
        if w in cue_heads and i >= 2:
            chunk = " ".join(t.text for t in toks[max(0, i - 2) : i + 1])
            proposals.append(chunk)
    # dedupe, keep short list
    seen = set()
    out = []
    for p in proposals:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out[:4]


# --- map annotated phrase strings to token spans -----------------------------
def phrases_to_spans(sentence: str, phrases: list[str]) -> list[Span]:
    """Locate each gold phrase as a contiguous token subsequence of the sentence.

    Matching is on normalised tokens, so casing/whitespace don't matter. Every
    occurrence of a phrase is returned (a phrase may repeat in a sentence).
    """
    toks = tokenize(sentence)
    norms = [t.norm for t in toks]
    spans: list[Span] = []
    for phrase in phrases:
        p = phrase.strip()
        if not p:
            continue
        ptoks = [t.norm for t in tokenize(p)]
        L = len(ptoks)
        if L == 0:
            continue
        for i in range(len(norms) - L + 1):
            if norms[i : i + L] == ptoks:
                spans.append(
                    Span(
                        start=i,
                        end=i + L,
                        text=" ".join(t.text for t in toks[i : i + L]),
                        score=1.0,
                    )
                )
    # de-duplicate identical spans
    uniq: dict[tuple[int, int], Span] = {}
    for s in spans:
        uniq[(s.start, s.end)] = s
    return sorted(uniq.values(), key=lambda s: s.start)


def parse_gold_annotations(df: pd.DataFrame) -> dict[int, list[str]]:
    """sent_uid -> list of gold phrase strings (from the 'gold_risk_terms' column)."""
    out: dict[int, list[str]] = {}
    for _, r in df.iterrows():
        raw = str(r.get("gold_risk_terms", "") or "")
        phrases = [p.strip() for p in raw.split(";") if p.strip()]
        out[int(r["sent_uid"])] = phrases
    return out


# --- evaluation against gold -------------------------------------------------
def _model_detect_sentence(
    sentence: str,
    model: RiskDetectorModel,
    cfg: Config,
    anchors: set[str],
) -> list[Span]:
    toks = tokenize(sentence)
    cands = candidate_spans(toks, anchors, cfg.max_phrase_len)
    scored = model.score_spans(cands)
    kept = [s for s in scored if s.score >= cfg.score_threshold]
    return nms(
        kept,
        iou_threshold=cfg.nms_iou_threshold,
        containment_threshold=cfg.nms_containment_threshold,
        length_bonus=cfg.phrase_length_bonus,
    )


def _lexicon_detect_sentence(sentence: str, entries: list[LexEntry]) -> list[Span]:
    return ground_truth_spans(tokenize(sentence), entries)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


@dataclass
class GoldEvaluation:
    n_sentences: int
    n_gold_spans: int
    model_vs_gold: dict
    lexicon_vs_gold: dict
    iou_threshold: float


def evaluate_against_gold(
    worksheet: pd.DataFrame,
    sentences: list[GoldSentence],
    model: RiskDetectorModel,
    cfg: Config,
    iou_threshold: float | None = None,
) -> GoldEvaluation:
    """Score the trained detector AND the lexicon labels against the gold spans."""
    thr = iou_threshold if iou_threshold is not None else cfg.match_iou_threshold
    entries = build_entries()
    anchors = anchor_tokens(entries)
    gold_map = parse_gold_annotations(worksheet)
    by_uid = {k: gs for k, gs in enumerate(sentences)}

    m_tp = m_fp = m_fn = 0
    l_tp = l_fp = l_fn = 0
    n_gold = 0
    for uid, phrases in gold_map.items():
        gs = by_uid.get(uid)
        if gs is None:
            continue
        gold_spans = phrases_to_spans(gs.text, phrases)
        n_gold += len(gold_spans)

        model_dets = _model_detect_sentence(gs.text, model, cfg, anchors)
        tp, fp, fn = match(model_dets, gold_spans, thr)
        m_tp += tp; m_fp += fp; m_fn += fn

        lex_dets = _lexicon_detect_sentence(gs.text, entries)
        tp, fp, fn = match(lex_dets, gold_spans, thr)
        l_tp += tp; l_fp += fp; l_fn += fn

    return GoldEvaluation(
        n_sentences=len(gold_map),
        n_gold_spans=n_gold,
        model_vs_gold=_prf(m_tp, m_fp, m_fn),
        lexicon_vs_gold=_prf(l_tp, l_fp, l_fn),
        iou_threshold=thr,
    )
