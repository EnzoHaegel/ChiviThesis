"""Sample sentences and write a gold-annotation worksheet.

Usage:
    python scripts/make_gold_worksheet.py --n 90 --out gold/worksheet.csv

The annotator then fills the `gold_risk_terms` column (risk phrases present in
the sentence, separated by ';') and saves it as gold/gold_annotations.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_iou.config import Config  # noqa: E402
from risk_iou.gold import sample_sentences, worksheet_dataframe  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=90)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = Config()
    sents = sample_sentences(cfg, n=args.n, seed=args.seed)
    df = worksheet_dataframe(sents, suggestions=True)

    out = args.out or str(Path(__file__).resolve().parents[1] / "gold" / "worksheet.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} sentences to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
