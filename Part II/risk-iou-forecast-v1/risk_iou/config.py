"""Central configuration for the risk-IoU pipeline.

All paths are resolved relative to the repository so the project is portable.
Tunables (thresholds, sampling caps) live here so a run is fully reproducible
from a single object that is also serialised into every report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Repository layout -------------------------------------------------------
# .../Part II/risk-iou-forecast-v1/risk_iou/config.py  -> REPO_ROOT = risk-iou-forecast-v1
REPO_ROOT = Path(__file__).resolve().parents[1]
PART_II_DIR = REPO_ROOT.parent                      # ".../Part II"
FINBERT_DIR = PART_II_DIR / "Finbert IoU"           # where the data already lives


@dataclass
class Config:
    # --- Data sources --------------------------------------------------------
    # Two folders of extracted "Risk Factors" .txt files (see project memory):
    #   - downloaded_risk_factors_all  : historical filings (2005-2020 ...)
    #   - downloaded_risk_factors_2025_2: recent filings incl. the 2025 block
    risk_dirs: tuple[str, ...] = (
        str(FINBERT_DIR / "downloaded_risk_factors_all"),
        str(FINBERT_DIR / "downloaded_risk_factors_2025_2"),
    )
    links_csv: str = str(FINBERT_DIR / "sec_10k_10q_links_2025_only_finbert.csv")

    # --- Temporal design -----------------------------------------------------
    # Everything strictly before `oos_year` is the "known"/historical corpus used
    # for training+testing (Phases 1-3). `oos_year` is the out-of-sample year
    # used for forecasting (Phase 4) — answering "which risks persist into 2025
    # vs. which are new in 2025".
    oos_year: int = 2025
    # Quarters the user explicitly wants as the 3-year training horizon. Files are
    # bucketed by filing quarter; if a quarter is sparse the pipeline reports it
    # rather than failing.
    train_quarters: tuple[str, ...] = (
        "2022Q1", "2022Q2", "2022Q3", "2022Q4",
        "2023Q1", "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    )
    # If the explicit train_quarters yield too few documents (the provided corpus
    # is dense in 2005-2020 and 2025 but sparse in 2022-2024), fall back to using
    # ALL filings before oos_year as the historical corpus. The provided
    # extraction only contains a handful of 2024Q3/Q4 filings inside the requested
    # window, which is far too sparse to define an "established" risk vocabulary,
    # so the threshold is set so the pipeline uses the full pre-2025 history (which
    # subsumes and extends the requested 3-year horizon) for a robust forecast.
    min_train_docs: int = 120

    # --- Sampling caps (keep a full run tractable & deterministic) -----------
    max_docs_per_group: int | None = 250   # cap historical / OOS doc counts; None = all
    max_tokens_per_doc: int = 4000          # truncate very long Risk Factors sections
    random_seed: int = 42

    # --- Span detection ------------------------------------------------------
    max_phrase_len: int = 5     # max token length of a candidate / lexicon phrase
    # A candidate span is "anchored": it must overlap at least one risk-anchor
    # token (a token that occurs in the lexicon vocabulary). This is the textual
    # analogue of region proposals around salient pixels.

    # --- IoU / NMS thresholds ------------------------------------------------
    nms_iou_threshold: float = 0.5       # suppress overlapping detections above this
    nms_containment_threshold: float = 0.8   # also suppress strongly-nested spans
    phrase_length_bonus: float = 0.02    # NMS tie-break favouring longer phrases
    # (kept small: a large bonus over-extends detections past the ground-truth
    # phrase and collapses IoU matching at strict thresholds)
    match_iou_threshold: float = 0.5     # detection<->ground-truth match (TP) cutoff
    eval_iou_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)

    # --- Model / split -------------------------------------------------------
    test_size: float = 0.20              # 80/20 train/test (grouped by filing)
    score_threshold: float = 0.50        # classifier prob cutoff for a positive detection

    # --- Forecasting (Phase 4) ----------------------------------------------
    # A historical risk term is considered "established" if it appears in at least
    # this fraction of historical quarters; a 2025 term is "novel" if it never
    # appeared (or appeared below the rarity floor) before oos_year.
    persistence_min_quarter_frac: float = 0.25
    novelty_max_hist_doc_freq: float = 0.01
    # Forecasting operates on a BOUNDED vocabulary so the analysis is robust:
    #  - only terms up to this many tokens (risk keywords are short; long unique
    #    strings would otherwise explode the term count and look spuriously novel);
    #  - a NOVEL term must also be *materially* disclosed in the OOS year (present
    #    in at least this fraction of OOS filings) — a one-off long phrase is not
    #    an emerging risk theme. Terms below the floor are tagged RARE, not novel.
    forecast_max_term_len: int = 3
    novelty_min_oos_doc_freq: float = 0.01

    # --- Outputs -------------------------------------------------------------
    output_dir: str = str(REPO_ROOT / "outputs")

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    return Config()
