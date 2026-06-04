"""Evaluate the trained detector and the lexicon labels against a gold set.

Usage:
    python scripts/eval_gold.py \
        --annotations gold/gold_annotations.csv \
        --model outputs/risk_detector_model.joblib

Re-derives the exact same sentence sample (same n/seed) the worksheet was built
from, so sent_uid lines up with the filled annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_iou.config import Config  # noqa: E402
from risk_iou.gold import evaluate_against_gold, sample_sentences  # noqa: E402
from risk_iou.model import RiskDetectorModel  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=str, default=str(REPO / "gold" / "gold_annotations.csv"))
    p.add_argument("--model", type=str, default=str(REPO / "outputs" / "risk_detector_model.joblib"))
    p.add_argument("--n", type=int, default=90)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default=str(REPO / "gold" / "gold_eval.json"))
    args = p.parse_args()

    cfg = Config()
    sents = sample_sentences(cfg, n=args.n, seed=args.seed)
    ws = pd.read_csv(args.annotations).fillna("")
    model = RiskDetectorModel.load(args.model)

    results = {}
    for thr in (0.3, 0.5, 0.7):
        ev = evaluate_against_gold(ws, sents, model, cfg, iou_threshold=thr)
        results[str(thr)] = {
            "model_vs_gold": ev.model_vs_gold,
            "lexicon_vs_gold": ev.lexicon_vs_gold,
        }
    payload = {
        "n_sentences": int(len(ws)),
        "n_gold_spans": int(sum(len(str(x).split(";")) if str(x).strip() else 0
                                for x in ws["gold_risk_terms"])),
        "by_iou_threshold": results,
        "note": "Gold = independent (lexicon-free) annotation. model_vs_gold is the "
                "'true' detector F1; lexicon_vs_gold validates the weak-supervision labels.",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
