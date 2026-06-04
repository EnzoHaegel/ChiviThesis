# Risk-IoU Forecast — v1

IoU/NMS-based **risk-keyword detection and out-of-sample forecasting** for SEC
*Risk Factors* disclosures (10-K / 10-Q).

The project transposes the **object-detection** evaluation toolkit — region
proposals, **Intersection-over-Union (IoU)**, **Non-Maximum Suppression (NMS)**
and **precision / recall / F1** — from 2-D image bounding boxes to **1-D token
spans** in financial-disclosure text. It then uses the trained detector to ask,
on the 2025 out-of-sample year:

> *Which risks **constantly exist** in 2025, and which **new** risk words were
> disclosed in 2025 that did not exist before?*

---

## The analogy (vision → text)

| Object detection | This project |
|---|---|
| Image | A filing's Risk-Factors document (token sequence) |
| Bounding box `[x0,y0,x1,y1]` | Token span `[start, end)` |
| Region proposals | Anchored n-gram windows around risk-vocabulary tokens |
| Detection score | Learned probability that a span is a risk keyword |
| IoU | 1-D interval IoU `|A∩B| / |A∪B|` |
| NMS | Greedy suppression of overlapping span detections |
| TP/FP/FN, P/R/F1 | Same, via IoU-matching detections to labeled spans |

---

## Pipeline (matches the spec phases)

**Phase 1 — Data processing & labeling** (`data.py`, `lexicon.py`, `spans.py`,
`labeling.py`)
- Parse filenames → `(date, form, cik, accession)`, bucket into calendar quarters.
- Split corpus into a **historical** partition (pre-2025) and the **2025
  out-of-sample** partition.
- Ground-truth risk spans via greedy longest-first lexicon matching across the
  canonical 10-K risk taxonomy (macro, market, credit/liquidity, operational,
  regulatory/legal, cyber, geopolitical/trade, health, climate/ESG).
- Anchored candidate spans (region proposals); each candidate is **labeled by
  IoU** against the ground truth. → `labeled_risk_dataset.csv`, `candidates_dataset.csv`.

**Phase 2 — Model development** (`model.py`)
- **80/20 train/test split grouped by filing** (no document leakage).
- Logistic-regression detector over char+word TF-IDF features of each span.

**Phase 3 — Optimization & evaluation** (`nms.py`, `evaluation.py`)
- Second-stage pipeline: *score → threshold → **NMS** → **IoU-match** to ground
  truth* → **TP/FP/FN → precision / recall / F1**, swept over IoU ∈ {0.3, 0.5, 0.7}.
- → `evaluation_report.json`, `nms_detections_test.csv`, `fig_prf_by_threshold.png`.

**Phase 4 — Out-of-sample forecasting** (`forecast.py`)
- Apply the trained detector to 2025; compare the 2025 risk vocabulary against
  the historical one.
- Classify each 2025 risk term as **PERSISTENT** (constantly exists),
  **INTERMITTENT**, **NOVEL** (new in 2025), or **RARE** (one-off, excluded).
- The forecast runs on a **bounded vocabulary** (terms ≤ 3 tokens) and a NOVEL
  term must be *materially* disclosed in 2025 (present in ≥ a min share of
  filings) — otherwise idiosyncratic long phrases would masquerade as new risks.
- Metrics: forecast recall (share of material 2025 terms seen before),
  historical-risk recall, vocabulary IoU, cross-quarter consistency, detection
  generalisation.
- → `forecast_2025_terms.csv`, `forecast_persistent.csv`, `forecast_novel.csv`,
  `forecast_summary.json`, figures.

---

## Results (full corpus: 1465 historical → 1181 OOS-2025 filings)

**Detection (held-out test, 281 filings, 10,887 ground-truth spans)**
- Candidate (window) level: accuracy **0.96**, F1 **0.83**.
- Span detection IoU@0.5 (standard operating point): F1 **0.60** (recall 0.96).
- IoU@0.7 (strict localisation): F1 0.17 — the honest ceiling of *weak-
  supervision* labels; the detected spans rarely line up token-perfectly with
  the lexicon ground truth. Report and read IoU@0.5 as the headline.

**2025 forecasting**
- 662 material risk terms → **469 persistent · 149 novel · 44 intermittent**
  (9,848 rare long-tail terms excluded).
- Forecast recall **0.77** (most 2025 risk language is recurring),
  historical-risk recall **0.92** (established risks reliably reappear).
- **New in 2025**: retaliatory / reciprocal tariffs, cybersecurity incidents,
  environmental-social (ESG), economic sanctions, geopolitical tensions,
  artificial intelligence, global supply chain — a faithful snapshot of the
  2025 disclosure landscape.

> **Caveat to report honestly:** ground truth is lexicon-based *weak
> supervision*, so F1 measures agreement with lexicon-derived labels, not an
> absolute gold standard. The strongest, lexicon-independent result is the
> Phase-4 emergence of new risk vocabulary.

### Gold-standard validation (`gold/`)

An independent 90-sentence gold set (annotated by reading, not by the lexicon)
quantifies the caveat above. Against the gold, span F1@0.5 is ≈ **0.34** for
both the detector and the raw lexicon — lower than the weak-supervision 0.60,
and **recall-limited**: the lexicon is precise (P≈0.58) but narrow (R≈0.24),
missing real risks outside the seed vocabulary (COVID-19, Dodd-Frank, GDPR,
material weakness, …). The fix is lexicon coverage / an embedding scorer, not
the IoU machinery. See `gold/GOLD_REPORT.md`. Reproduce:

```bash
python scripts/make_gold_worksheet.py            # -> gold/worksheet.csv
python gold/seed_gold_llm.py                     # -> gold/gold_annotations.csv (LLM draft)
python scripts/eval_gold.py                       # -> gold/gold_eval.json
```

---

## Install & run

```bash
pip install -r requirements.txt

# full run (default sampling cap of 250 docs per partition)
python scripts/run_pipeline.py

# use every available document
python scripts/run_pipeline.py --all-docs

# smaller / faster
python scripts/run_pipeline.py --max-docs 150
```

All deliverables are written to `outputs/`. A `REPORT.md` summarises the F1 /
recall evaluation and the 2025 forecasting analysis.

## Tests

```bash
python -m pytest -q
```

Covers IoU, NMS, IoU-matching & F1 math, filename/quarter parsing, tokeniser,
lexicon, and a full end-to-end smoke test on a synthetic corpus.

## Configuration

Every tunable (thresholds, sampling caps, temporal design, output dir) lives in
`risk_iou/config.py` and is serialised into `outputs/run_manifest.json` for
reproducibility.

## Notes on the data

The provided corpus is dense in 2005–2020 and in 2025 but sparse in 2022–2024.
The pipeline first tries the explicit 2022Q1–2024Q4 horizon; if it yields too
few filings it transparently falls back to *all* pre-2025 filings as the
historical corpus and records which choice was made in the manifest.

The enhanced semantic backend (FinBERT / sentence-transformers) is **optional**:
if those libraries are absent the detector uses TF-IDF features and the pipeline
runs identically.
