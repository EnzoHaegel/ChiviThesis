"""Corpus loading, filename parsing and temporal splits.

Risk-Factors files are named like::

    2024-07-11_10-K_edgar_data_0000023217_0001558370-24-009764.txt
    <filing_date>_<form>_edgar_data_<cik>_<accession_number>.txt

We parse those fields, bucket each filing into a calendar quarter, and build the
three logical partitions the spec asks for:

  * historical (pre-OOS-year) corpus  -> Phases 1-3 (train/test)
  * out-of-sample year (default 2025) -> Phase 4 (forecasting)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config

_FNAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<form>10-[KQ])_"
    r"edgar_data_(?P<cik>\d+)_"
    r"(?P<accession>[\d-]+)\.txt$"
)


@dataclass
class Filing:
    path: str
    date: str            # YYYY-MM-DD
    year: int
    quarter: str         # e.g. "2024Q3"
    form: str            # 10-K / 10-Q
    cik: str
    accession: str

    @property
    def filing_id(self) -> str:
        return f"{self.cik}_{self.accession}"


def _quarter(date: str) -> tuple[int, str]:
    y, m, _ = date.split("-")
    year = int(y)
    q = (int(m) - 1) // 3 + 1
    return year, f"{year}Q{q}"


def parse_filename(name: str) -> dict | None:
    """Parse a Risk-Factors filename → field dict, or None if it doesn't match."""
    m = _FNAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    year, quarter = _quarter(d["date"])
    d["year"] = year
    d["quarter"] = quarter
    return d


def discover_filings(cfg: Config) -> list[Filing]:
    """Scan the configured risk directories and return parsed Filing records.

    The same accession can appear in both folders; we keep the first occurrence.
    """
    seen: set[str] = set()
    filings: list[Filing] = []
    for d in cfg.risk_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".txt"):
                continue
            parsed = parse_filename(name)
            if parsed is None:
                continue
            key = f"{parsed['cik']}_{parsed['accession']}"
            if key in seen:
                continue
            seen.add(key)
            filings.append(
                Filing(
                    path=str(Path(d) / name),
                    date=parsed["date"],
                    year=parsed["year"],
                    quarter=parsed["quarter"],
                    form=parsed["form"],
                    cik=parsed["cik"],
                    accession=parsed["accession"],
                )
            )
    return filings


def read_text(filing: Filing, max_chars: int = 400_000) -> str:
    """Read a filing's Risk-Factors text (utf-8, tolerant of bad bytes)."""
    try:
        with open(filing.path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(max_chars)
    except OSError:
        return ""


@dataclass
class CorpusSplit:
    historical: list[Filing]   # pre-OOS year (training + testing material)
    oos: list[Filing]          # out-of-sample year
    train_quarter_hits: dict[str, int]
    used_explicit_quarters: bool


def _sample(filings: list[Filing], cap: int | None, seed: int) -> list[Filing]:
    if cap is None or len(filings) <= cap:
        return filings
    import random

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(filings)), cap))
    return [filings[i] for i in idx]


def build_split(cfg: Config, filings: list[Filing] | None = None) -> CorpusSplit:
    """Partition the corpus into historical vs out-of-sample.

    Tries the explicit `train_quarters` (2022Q1..2024Q4) first; if that yields
    fewer than `min_train_docs` filings (the provided corpus is sparse there),
    it transparently falls back to *all* pre-OOS-year filings and records that.
    """
    if filings is None:
        filings = discover_filings(cfg)

    quarter_set = set(cfg.train_quarters)
    explicit = [f for f in filings if f.quarter in quarter_set]
    train_hits: dict[str, int] = {q: 0 for q in cfg.train_quarters}
    for f in explicit:
        train_hits[f.quarter] += 1

    if len(explicit) >= cfg.min_train_docs:
        historical = explicit
        used_explicit = True
    else:
        historical = [f for f in filings if f.year < cfg.oos_year]
        used_explicit = False

    oos = [f for f in filings if f.year == cfg.oos_year]

    historical = _sample(historical, cfg.max_docs_per_group, cfg.random_seed)
    oos = _sample(oos, cfg.max_docs_per_group, cfg.random_seed + 1)

    return CorpusSplit(
        historical=historical,
        oos=oos,
        train_quarter_hits=train_hits,
        used_explicit_quarters=used_explicit,
    )
