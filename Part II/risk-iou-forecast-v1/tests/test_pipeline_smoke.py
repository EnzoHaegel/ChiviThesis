"""End-to-end smoke test on a tiny synthetic corpus.

Builds a handful of fake Risk-Factors files (correct filename convention) with
known risk phrases, runs the whole Phase 1-4 pipeline, and asserts the
deliverables are produced and internally consistent.
"""
import json
import os

from risk_iou.config import Config
from risk_iou.pipeline import run


HIST_TEXT = (
    "Our business is subject to inflation and rising interest rates. "
    "Supply chain disruption and a labor shortage could harm operations. "
    "We face cybersecurity threats and a possible data breach. "
    "Litigation and regulatory compliance remain ongoing risks. "
    "Market volatility and liquidity risk affect our financing. "
    "An economic recession would reduce consumer spending materially."
)

OOS_TEXT = (
    "Inflation and rising interest rates continue to pressure margins. "
    "Supply chain disruption persists alongside a labor shortage. "
    "New tariffs and a trade war create geopolitical uncertainty. "
    "Cybersecurity threats including ransomware grow each year. "
    "Artificial intelligence introduces software vulnerability concerns. "
    "Market volatility and liquidity risk remain material to us."
)


def _make_corpus(root):
    hist_dir = os.path.join(root, "hist")
    oos_dir = os.path.join(root, "oos")
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(oos_dir, exist_ok=True)

    # 14 historical filings across 2018-2020, 7 ciks (so a grouped split has
    # enough groups), and 6 out-of-sample 2025 filings across two quarters.
    for i in range(14):
        year = 2018 + (i % 3)
        cik = f"{1000 + (i % 7):010d}"
        acc = f"000000000{i:02d}-18-000001"
        name = f"{year}-03-15_10-K_edgar_data_{cik}_{acc}.txt"
        with open(os.path.join(hist_dir, name), "w", encoding="utf-8") as fh:
            fh.write(HIST_TEXT)

    for i in range(6):
        q_month = "03" if i % 2 == 0 else "08"
        cik = f"{2000 + i:010d}"
        acc = f"000000001{i:02d}-25-000001"
        name = f"2025-{q_month}-10_10-K_edgar_data_{cik}_{acc}.txt"
        with open(os.path.join(oos_dir, name), "w", encoding="utf-8") as fh:
            fh.write(OOS_TEXT)

    return hist_dir, oos_dir


def test_pipeline_end_to_end(tmp_path):
    hist_dir, oos_dir = _make_corpus(str(tmp_path))
    out = os.path.join(str(tmp_path), "outputs")
    cfg = Config(
        risk_dirs=(hist_dir, oos_dir),
        output_dir=out,
        max_docs_per_group=None,
        min_train_docs=5,
        oos_year=2025,
    )

    artifacts = run(cfg)

    # deliverables exist
    expected = [
        "labeled_risk_dataset.csv",
        "candidates_dataset.csv",
        "risk_detector_model.joblib",
        "evaluation_report.json",
        "nms_detections_test.csv",
        "forecast_2025_terms.csv",
        "forecast_persistent.csv",
        "forecast_novel.csv",
        "forecast_summary.json",
        "run_manifest.json",
        "REPORT.md",
        "fig_prf_by_threshold.png",
        "fig_status_breakdown.png",
    ]
    for name in expected:
        path = os.path.join(out, name)
        assert os.path.exists(path), f"missing deliverable: {name}"
        assert os.path.getsize(path) > 0

    # metrics are well-formed
    ev = artifacts.evaluation
    assert ev["n_test_documents"] >= 1
    for thr, m in ev["detection_by_iou_threshold"].items():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0

    # forecasting found the OOS-only risks as novel (tariffs / trade war / AI)
    fc = artifacts.forecast_summary
    assert fc["n_oos_terms_material"] > 0
    assert 0.0 <= fc["forecast_recall_terms_seen_before"] <= 1.0
    novel_path = os.path.join(out, "forecast_novel.csv")
    novel_txt = open(novel_path, encoding="utf-8").read().lower()
    # at least one of the genuinely-new 2025 terms should be flagged novel
    assert any(k in novel_txt for k in ["tariff", "trade war", "artificial intelligence", "ransomware"])

    # manifest round-trips as JSON
    with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    assert man["corpus"]["n_oos"] == 6
