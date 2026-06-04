"""Phase 4 — out-of-sample (2025) forecasting.

The trained detector is applied to the held-out OOS year. We answer the core
research question — *which risks constantly exist in 2025 vs. which are newly
disclosed* — by comparing the OOS risk vocabulary against the historical one:

  * PERSISTENT : established historically (present in a sizeable share of past
    quarters) and still detected in 2025  -> "constantly exists".
  * NOVEL      : detected in 2025 but essentially unseen before                -> "new risk word in 2025".
  * INTERMITTENT: seen before but only sporadically.

Reported metrics:
  * forecast recall = share of 2025 risk terms that were predictable from history
    (i.e. 1 - novel share)  -> how well the past forecasts 2025;
  * historical-risk recall = share of established historical risks that reappear
    in 2025  -> prediction consistency of persistent risks;
  * vocabulary IoU between the historical and 2025 risk-term sets;
  * cross-quarter consistency = mean pairwise IoU of 2025 quarterly risk sets;
  * detection generalisation = the Phase-3 detection F1/recall recomputed on 2025.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .evaluation import detect_document, evaluate
from .labeling import DocAnnotation
from .lexicon import LexEntry, build_entries
from .model import RiskDetectorModel


def resolve_category(term_norm: str, entries: list[LexEntry]) -> str:
    """Map a detected term to a lexicon category (longest matching phrase wins)."""
    toks = term_norm.split()
    tokset = set(toks)
    best_cat = "other"
    best_len = 0
    for e in entries:  # entries are longest-first
        if len(e.tokens) > len(toks):
            continue
        if all(t in tokset for t in e.tokens) and len(e.tokens) > best_len:
            best_cat = e.category
            best_len = len(e.tokens)
    return best_cat


def _detections_table(
    annotations: list[DocAnnotation],
    model: RiskDetectorModel,
    cfg: Config,
    entries: list[LexEntry],
) -> pd.DataFrame:
    """One row per (doc, detected term)."""
    rows = []
    for ann in annotations:
        for d in detect_document(ann, model, cfg):
            term = " ".join(d.text.lower().split())
            if not term:
                continue
            # bound the forecast vocabulary to short risk keywords; long unique
            # strings explode the term count and masquerade as novel risks.
            if len(term.split()) > cfg.forecast_max_term_len:
                continue
            rows.append(
                {
                    "filing_id": ann.filing.filing_id,
                    "quarter": ann.filing.quarter,
                    "term": term,
                    "category": resolve_category(term, entries),
                    "score": d.score,
                }
            )
    return pd.DataFrame(rows)


def _term_profile(det: pd.DataFrame, n_docs: int, quarters: list[str]) -> pd.DataFrame:
    """Aggregate detections to per-term frequencies and quarter coverage."""
    if det.empty:
        return pd.DataFrame(
            columns=["term", "category", "doc_freq", "n_docs", "quarter_frac", "n_quarters"]
        )
    n_q = max(len(set(quarters)), 1)
    grp = det.groupby("term")
    out = grp.agg(
        category=("category", lambda s: s.mode().iat[0] if not s.mode().empty else "other"),
        n_docs=("filing_id", "nunique"),
        n_quarters=("quarter", "nunique"),
    ).reset_index()
    out["doc_freq"] = out["n_docs"] / max(n_docs, 1)
    out["quarter_frac"] = out["n_quarters"] / n_q
    return out


@dataclass
class ForecastResult:
    term_table: pd.DataFrame          # per-2025-term status vs history
    summary: dict
    persistent: pd.DataFrame
    novel: pd.DataFrame


def _set_iou(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def forecast(
    historical: list[DocAnnotation],
    oos: list[DocAnnotation],
    model: RiskDetectorModel,
    cfg: Config,
    entries: list[LexEntry] | None = None,
) -> ForecastResult:
    if entries is None:
        entries = build_entries()

    hist_q = [a.filing.quarter for a in historical]
    oos_q = [a.filing.quarter for a in oos]

    hist_det = _detections_table(historical, model, cfg, entries)
    oos_det = _detections_table(oos, model, cfg, entries)

    hist_prof = _term_profile(hist_det, len(historical), hist_q).set_index("term")
    oos_prof = _term_profile(oos_det, len(oos), oos_q)

    # Classify every term that appears in 2025.
    records = []
    for _, r in oos_prof.iterrows():
        term = r["term"]
        h = hist_prof.loc[term] if term in hist_prof.index else None
        h_doc_freq = float(h["doc_freq"]) if h is not None else 0.0
        h_q_frac = float(h["quarter_frac"]) if h is not None else 0.0
        oos_doc_freq = float(r["doc_freq"])
        established = h_q_frac >= cfg.persistence_min_quarter_frac
        unseen = h_doc_freq <= cfg.novelty_max_hist_doc_freq
        material = oos_doc_freq >= cfg.novelty_min_oos_doc_freq
        if established:
            # seen across many past quarters AND still present in 2025
            status = "PERSISTENT"
        elif unseen and material:
            # essentially absent before, yet broadly disclosed in 2025
            status = "NOVEL"
        elif not material:
            # too idiosyncratic in 2025 to call an emerging theme
            status = "RARE"
        else:
            status = "INTERMITTENT"
        records.append(
            {
                "term": term,
                "category": r["category"],
                "oos_doc_freq": round(float(r["doc_freq"]), 4),
                "oos_n_docs": int(r["n_docs"]),
                "hist_doc_freq": round(h_doc_freq, 4),
                "hist_quarter_frac": round(h_q_frac, 4),
                "status": status,
            }
        )
    term_table = pd.DataFrame(records).sort_values(
        ["status", "oos_doc_freq"], ascending=[True, False]
    ).reset_index(drop=True)

    persistent = term_table[term_table["status"] == "PERSISTENT"].reset_index(drop=True)
    novel = term_table[term_table["status"] == "NOVEL"].reset_index(drop=True)

    # --- set-level metrics ---------------------------------------------------
    # Restrict the headline analysis to the MATERIAL vocabulary (terms disclosed
    # in a meaningful share of 2025 filings); the rare long-tail is reported
    # separately and excluded from the recall denominator so the metric reflects
    # genuine emerging-vs-persistent themes, not idiosyncratic one-offs.
    material_tbl = term_table[term_table["status"] != "RARE"]
    hist_terms = set(hist_prof.index)
    oos_terms = set(material_tbl["term"])
    established = set(
        hist_prof[hist_prof["quarter_frac"] >= cfg.persistence_min_quarter_frac].index
    )

    n_material = max(len(material_tbl), 1)
    n_novel = int((term_table["status"] == "NOVEL").sum())
    novel_share = n_novel / n_material
    forecast_recall = 1.0 - novel_share          # share of material 2025 terms seen before
    hist_risk_recall = (
        len(established & oos_terms) / len(established) if established else 0.0
    )
    vocab_iou = _set_iou(hist_terms, oos_terms)

    # cross-quarter consistency (mean pairwise IoU of 2025 quarterly term sets)
    consistency = _cross_quarter_consistency(oos_det)

    # detection generalisation on 2025 (Phase-3 metrics recomputed OOS)
    gen = evaluate(oos, model, cfg, collect_detections=False)
    gen_at_match = gen.by_threshold.get(cfg.match_iou_threshold)

    summary = {
        "n_historical_docs": len(historical),
        "n_oos_docs": len(oos),
        "n_oos_terms_total": int(len(oos_prof)),
        "n_oos_terms_material": int(len(material_tbl)),
        "n_persistent": int((term_table["status"] == "PERSISTENT").sum()),
        "n_intermittent": int((term_table["status"] == "INTERMITTENT").sum()),
        "n_novel": n_novel,
        "n_rare_excluded": int((term_table["status"] == "RARE").sum()),
        "forecast_recall_terms_seen_before": round(forecast_recall, 4),
        "novel_share": round(novel_share, 4),
        "historical_risk_recall": round(hist_risk_recall, 4),
        "vocabulary_iou_hist_vs_2025": round(vocab_iou, 4),
        "cross_quarter_consistency_iou": round(consistency, 4),
        "detection_generalisation": gen_at_match,
    }

    return ForecastResult(
        term_table=term_table, summary=summary, persistent=persistent, novel=novel
    )


def _cross_quarter_consistency(oos_det: pd.DataFrame) -> float:
    if oos_det.empty:
        return 0.0
    sets = {
        q: set(g["term"]) for q, g in oos_det.groupby("quarter")
    }
    quarters = sorted(sets)
    if len(quarters) < 2:
        return 1.0 if quarters else 0.0
    ious = []
    for i in range(len(quarters)):
        for j in range(i + 1, len(quarters)):
            ious.append(_set_iou(sets[quarters[i]], sets[quarters[j]]))
    return sum(ious) / len(ious) if ious else 0.0
