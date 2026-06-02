"""End-to-end orchestrator: Phase 1 -> Phase 4, writing every deliverable."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import pandas as pd

from . import plots
from .config import Config, default_config
from .data import build_split, discover_filings
from .evaluation import evaluate
from .labeling import (
    annotate_corpus,
    candidates_dataframe,
    ground_truth_dataframe,
)
from .lexicon import anchor_tokens, build_entries
from .model import RiskDetectorModel, grouped_train_test_split
from .forecast import forecast


def _log(msg: str) -> None:
    print(f"[risk_iou] {msg}", flush=True)


@dataclass
class PipelineArtifacts:
    manifest: dict
    evaluation: dict
    forecast_summary: dict


def run(cfg: Config | None = None) -> PipelineArtifacts:
    cfg = cfg or default_config()
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # --- Corpus & split ------------------------------------------------------
    _log("discovering filings ...")
    filings = discover_filings(cfg)
    split = build_split(cfg, filings)
    _log(
        f"{len(filings)} filings parsed | historical={len(split.historical)} "
        f"oos({cfg.oos_year})={len(split.oos)} | "
        f"explicit_quarters_used={split.used_explicit_quarters}"
    )

    entries = build_entries()
    anchors = anchor_tokens(entries)

    # --- Phase 1: annotate + label ------------------------------------------
    _log("Phase 1: annotating historical corpus (IoU labeling) ...")
    hist_ann = annotate_corpus(split.historical, cfg, entries, anchors)
    _log("Phase 1: annotating out-of-sample corpus ...")
    oos_ann = annotate_corpus(split.oos, cfg, entries, anchors)

    cand_df = candidates_dataframe(hist_ann)
    gt_df = ground_truth_dataframe(hist_ann)
    gt_df.to_csv(os.path.join(out, "labeled_risk_dataset.csv"), index=False)
    cand_df.to_csv(os.path.join(out, "candidates_dataset.csv"), index=False)
    _log(
        f"Phase 1: {len(gt_df)} ground-truth risk spans, "
        f"{len(cand_df)} candidate windows "
        f"({int(cand_df['label'].sum()) if len(cand_df) else 0} positive)"
    )

    if cand_df.empty or cand_df["label"].nunique() < 2:
        raise RuntimeError(
            "Not enough labeled data to train (need both classes). "
            "Check that the risk directories contain text files."
        )

    # --- Phase 2: train/test split + model ----------------------------------
    _log("Phase 2: 80/20 grouped split + model training ...")
    sr = grouped_train_test_split(cand_df, cfg.test_size, cfg.random_seed)
    train_df = cand_df.iloc[sr.train_idx].reset_index(drop=True)
    test_df = cand_df.iloc[sr.test_idx].reset_index(drop=True)
    test_filing_ids = set(test_df["filing_id"])
    _log(
        f"Phase 2: train={len(train_df)} rows / test={len(test_df)} rows | "
        f"test filings={len(test_filing_ids)}"
    )

    model = RiskDetectorModel(seed=cfg.random_seed).fit(train_df)
    model.save(os.path.join(out, "risk_detector_model.joblib"))

    # --- Phase 3: second-stage IoU + NMS + F1 evaluation --------------------
    _log("Phase 3: NMS + IoU evaluation on held-out test filings ...")
    test_ann = [a for a in hist_ann if a.filing.filing_id in test_filing_ids]
    eval_res = evaluate(test_ann, model, cfg)
    eval_res.detections_df.to_csv(
        os.path.join(out, "nms_detections_test.csv"), index=False
    )

    eval_payload = {
        "n_test_documents": eval_res.n_documents,
        "n_detections_after_nms": eval_res.n_detections,
        "n_ground_truth_spans": eval_res.n_ground_truth,
        "candidate_level": eval_res.candidate_accuracy,
        "detection_by_iou_threshold": {
            str(t): m for t, m in eval_res.by_threshold.items()
        },
    }
    with open(os.path.join(out, "evaluation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(eval_payload, fh, indent=2)
    _log(
        "Phase 3: F1@%.2f = %s"
        % (
            cfg.match_iou_threshold,
            eval_res.by_threshold.get(cfg.match_iou_threshold, {}).get("f1"),
        )
    )

    # --- Phase 4: out-of-sample forecasting ---------------------------------
    _log("Phase 4: 2025 out-of-sample forecasting ...")
    fc = forecast(hist_ann, oos_ann, model, cfg, entries)
    fc.term_table.to_csv(os.path.join(out, "forecast_2025_terms.csv"), index=False)
    fc.persistent.to_csv(os.path.join(out, "forecast_persistent.csv"), index=False)
    fc.novel.to_csv(os.path.join(out, "forecast_novel.csv"), index=False)
    with open(os.path.join(out, "forecast_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(fc.summary, fh, indent=2)

    # --- Plots ---------------------------------------------------------------
    _log("rendering figures ...")
    plots.plot_prf_by_threshold(
        eval_res.by_threshold, os.path.join(out, "fig_prf_by_threshold.png")
    )
    plots.plot_status_breakdown(fc.summary, os.path.join(out, "fig_status_breakdown.png"))
    plots.plot_top_terms(
        fc.novel, "oos_doc_freq", "Top novel risk terms in 2025",
        os.path.join(out, "fig_top_novel.png"),
    )
    plots.plot_top_terms(
        fc.persistent, "oos_doc_freq", "Top persistent risk terms in 2025",
        os.path.join(out, "fig_top_persistent.png"),
    )
    plots.plot_category_distribution(
        fc.term_table, os.path.join(out, "fig_category_status.png")
    )

    # --- Manifest + markdown report -----------------------------------------
    manifest = {
        "version": "1.0.0",
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
        "corpus": {
            "n_filings_total": len(filings),
            "n_historical": len(split.historical),
            "n_oos": len(split.oos),
            "explicit_quarters_used": split.used_explicit_quarters,
            "train_quarter_hits": split.train_quarter_hits,
        },
    }
    with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    _write_markdown_report(out, manifest, eval_payload, fc.summary, fc)
    _log(f"DONE in {manifest['elapsed_seconds']}s. Deliverables in: {out}")

    return PipelineArtifacts(
        manifest=manifest, evaluation=eval_payload, forecast_summary=fc.summary
    )


def _write_markdown_report(out, manifest, eval_payload, fc_summary, fc) -> None:
    m = eval_payload["detection_by_iou_threshold"]
    lines = []
    lines.append("# Risk-IoU v1 — F1-score & Recall Evaluation Report\n")
    lines.append("## Corpus\n")
    c = manifest["corpus"]
    lines.append(f"- Total filings parsed: **{c['n_filings_total']}**")
    lines.append(f"- Historical (train/test): **{c['n_historical']}**")
    lines.append(f"- Out-of-sample {manifest['config']['oos_year']}: **{c['n_oos']}**")
    lines.append(f"- Explicit 2022-2024 quarters used as horizon: **{c['explicit_quarters_used']}**\n")

    lines.append("## Phase 3 — detection metrics (held-out test filings)\n")
    lines.append("Candidate (window) level:")
    cl = eval_payload["candidate_level"]
    lines.append(
        f"- accuracy={cl['accuracy']} precision={cl['precision']} "
        f"recall={cl['recall']} f1={cl['f1']}\n"
    )
    lines.append("Span detection (NMS + IoU matching):\n")
    lines.append("| IoU thr | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|--------:|---:|---:|---:|----------:|-------:|---:|")
    for t in sorted(m, key=float):
        r = m[t]
        lines.append(
            f"| {t} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']} | {r['recall']} | {r['f1']} |"
        )
    lines.append("")

    lines.append("## Phase 4 — 2025 out-of-sample forecasting\n")
    s = fc_summary
    lines.append(f"- 2025 distinct risk terms detected: **{s['n_oos_terms']}**")
    lines.append(f"- Persistent (constantly exist): **{s['n_persistent']}**")
    lines.append(f"- Intermittent: **{s['n_intermittent']}**")
    lines.append(f"- Novel (new in 2025): **{s['n_novel']}**")
    lines.append(
        f"- Forecast recall (2025 terms seen before): "
        f"**{s['forecast_recall_terms_seen_before']}**"
    )
    lines.append(f"- Historical-risk recall (established risks reappearing in 2025): **{s['historical_risk_recall']}**")
    lines.append(f"- Vocabulary IoU (history vs 2025): **{s['vocabulary_iou_hist_vs_2025']}**")
    lines.append(f"- Cross-quarter consistency (IoU): **{s['cross_quarter_consistency_iou']}**")
    gen = s.get("detection_generalisation") or {}
    if gen:
        lines.append(
            f"- Detection generalisation on 2025 (IoU@match): "
            f"precision={gen.get('precision')} recall={gen.get('recall')} f1={gen.get('f1')}"
        )
    lines.append("")
    if not fc.novel.empty:
        lines.append("### Top new risk terms disclosed in 2025\n")
        for _, r in fc.novel.head(15).iterrows():
            lines.append(f"- `{r['term']}` ({r['category']}) — in {r['oos_doc_freq']:.1%} of 2025 filings")
        lines.append("")
    if not fc.persistent.empty:
        lines.append("### Top constantly-present risk terms\n")
        for _, r in fc.persistent.head(15).iterrows():
            lines.append(f"- `{r['term']}` ({r['category']}) — 2025 freq {r['oos_doc_freq']:.1%}, hist quarters {r['hist_quarter_frac']:.1%}")
        lines.append("")

    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
