# Gold-standard validation (independent of the lexicon)

**Why:** the Phase-3 metrics (span F1@0.5 ≈ 0.60) are computed against
*weak-supervision* labels derived from the seed lexicon, so they partly measure
self-consistency. This report confronts both the trained detector and the
lexicon labels with an **independent** gold annotation — risk phrases marked by
reading the text, not by the lexicon.

**Protocol:** 90 sentences sampled stratified by year (`scripts/make_gold_worksheet.py`,
seed 7). An annotator marked the genuine risk-keyword/-driver phrases per
sentence (boilerplate left empty). 52/90 sentences carry ≥1 risk term; **89 gold
phrases** total. The detector and the lexicon are scored against the gold with
the same IoU matching as Phase 3 (`scripts/eval_gold.py`).

> The labels here are a **model-assisted (LLM) gold draft**, independent of the
> seed lexicon but **not yet human-reviewed**. The worksheet
> (`gold/worksheet.csv` → `gold/gold_annotations.csv`) is built so Enzo can
> adjust the labels for a thesis-grade number. Sample is small (89 phrases), so
> read these as indicative.

## Results

| IoU | Model vs gold (P / R / F1) | Lexicon vs gold (P / R / F1) |
|----:|:--------------------------:|:----------------------------:|
| 0.3 | 0.36 / 0.35 / **0.36** | 0.63 / 0.26 / **0.37** |
| 0.5 | 0.34 / 0.33 / **0.34** | 0.58 / 0.24 / **0.34** |
| 0.7 | 0.18 / 0.17 / 0.17 | 0.39 / 0.16 / 0.23 |

## Reading

- **The honest number is lower than the weak-supervision F1.** Against
  independent labels the detector scores ~**0.34 F1 @IoU0.5**, versus 0.60 on the
  lexicon-derived test set. The 0.60 was optimistic because both the labels and
  the model share the lexicon's vocabulary.
- **The bottleneck is recall, not precision.** The lexicon is *precise* (P≈0.58:
  when it fires it is usually a real risk term) but *narrow* (R≈0.24): it misses
  most independently-annotated risks because they fall outside the seed
  vocabulary — e.g. *COVID-19, Dodd-Frank Act, GDPR, material weakness, natural
  gas prices, redemption requests, strike, LIFO method, GHG, FAA, collective
  bargaining agreements, AAA ratings*.
- **The learned model extends coverage modestly.** Its recall (0.33) exceeds the
  raw lexicon's (0.24) — it generalises a little beyond the seed terms via
  character n-grams — but at lower precision (0.34 vs 0.58).

## Implication for the thesis

The IoU/NMS detection machinery is sound; the limiting factor is **lexicon
coverage**. To raise the true F1 (priority order):
1. **Expand the lexicon** to the gaps surfaced above, or seed it from a finance
   risk ontology; re-measure on this *held-out* gold (do **not** tune the lexicon
   to the gold — keep it as an honest test set).
2. Add an **embedding / FinBERT** scorer so detection generalises beyond exact
   seed terms.
3. Have a human review the ~90 labels (some calls — *demand, supply,
   subscription* — are debatable) and enlarge the gold to ~300 sentences for
   tighter confidence intervals.

The Phase-4 forecasting result (emergence of tariffs / AI / ESG as new 2025
risks) is unaffected and remains the strongest, lexicon-robust finding.
