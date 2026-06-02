"""CLI entry point: run the full Phase 1-4 risk-IoU pipeline.

Usage (from the repo root):
    python -m scripts.run_pipeline
    python scripts/run_pipeline.py --max-docs 150 --oos-year 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_iou.config import Config  # noqa: E402
from risk_iou.pipeline import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Risk-IoU forecasting pipeline")
    p.add_argument("--max-docs", type=int, default=None,
                   help="cap docs per group (historical / oos). Omit for config default.")
    p.add_argument("--all-docs", action="store_true",
                   help="use every document (no sampling cap)")
    p.add_argument("--oos-year", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.all_docs:
        cfg.max_docs_per_group = None
    elif args.max_docs is not None:
        cfg.max_docs_per_group = args.max_docs
    if args.oos_year is not None:
        cfg.oos_year = args.oos_year
    if args.max_tokens is not None:
        cfg.max_tokens_per_doc = args.max_tokens
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
