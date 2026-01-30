# Bond Direction Prediction Model V2
## Multi-Horizon Classification (D+1 to D+10)

## Overview

### Objective
This model predicts the **future direction** (UP/DOWN/NEUTRAL) of bond returns from Risk Factors text, for **multiple time horizons after the earnings announcement**.

### Why Multi-Horizon?

```
                     Earnings Announcement (rdq)
                              │
     Risk Factors             │
    published BEFORE ◄────────┤
                              │
                              ▼
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ D-1 │ D+0 │ D+1 │ D+2 │ D+3 │ D+4 │ D+5 │ D+6 │ D+7 │ D+8 │ D+10│
    └─────┴─────┴──┬──┴──┬──┴─────┴─────┴──┬──┴─────┴─────┴─────┴──┬──┘
                   │     │                 │                       │
                   ▼     ▼                 ▼                       ▼
                 D+1   D+2               D+5                     D+10
                   │     │                 │                       │
                   └─────┴────────┬────────┴───────────────────────┘
                                  │
                            PREDICTIONS
                          (UP/DOWN/NEUTRAL)
```

**T+0 = announcement day** → Not really a prediction, it's simultaneous!  
**D+1 to D+10** → These are real **future predictions** ✅

---

## Predicted Horizons

| Horizon | Description | Interest |
|---------|-------------|----------|
| **D+1** | Return 1 day after announcement | Immediate market reaction |
| **D+2** | Return 2 days after | Trend confirmation |
| **D+5** | Return 5 days after | Medium-term effect (1 week) |
| **D+10** | Return 10 days after | Long-term effect (2 weeks) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUT                                      │
│              Risk Factors Text (max 256 tokens)                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FinBERT ENCODER                                 │
│  12 Transformer layers (Frozen: 1-11, Trainable: 12)                │
│  Output: [CLS] token [batch, 768]                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SHARED LAYER                                    │
│  Linear(768 → 256) → LayerNorm → GELU → Dropout(0.2)                │
│  Output: shared features [batch, 256]                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   HEAD D+1        │ │   HEAD D+2        │ │   HEAD D+5        │ ...
│ Linear(256→64)    │ │ Linear(256→64)    │ │ Linear(256→64)    │
│ GELU → Dropout    │ │ GELU → Dropout    │ │ GELU → Dropout    │
│ Linear(64→3)      │ │ Linear(64→3)      │ │ Linear(64→3)      │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
          │                     │                     │
          ▼                     ▼                     ▼
    [P(↓), P(→), P(↑)]   [P(↓), P(→), P(↑)]   [P(↓), P(→), P(↑)]
       for D+1               for D+2               for D+5
```

### Multi-Task Learning

The model shares FinBERT representations and the hidden layer across all horizons, then uses **specialized heads** for each horizon.

**Advantages:**
- Implicit regularization (shared representations)
- More memory efficient than training 4 separate models
- Captures patterns common to all horizons

---

## Return Classification

### Direction Thresholds

```
        DOWN              NEUTRAL              UP
    ◄──────────────►◄─────────────────►◄──────────────►
         < -0.5%      -0.5% to +0.5%       > +0.5%
```

| Class | Condition | Code |
|-------|-----------|------|
| DOWN | return < -0.5% | 0 |
| NEUTRAL | -0.5% ≤ return ≤ +0.5% | 1 |
| UP | return > +0.5% | 2 |

---

## Missing Data Handling

Some observations don't have all horizons available. The model uses a **loss mask**:

```python
# For each observation
labels = [class_D1, class_D2, class_D5, class_D10]
masks = [1, 1, 0, 1]  # 0 = missing data

# Loss computed only on valid horizons
loss = masked_cross_entropy(logits, labels, masks)
```

---

## Evaluation Metrics

### Per Horizon
- **Accuracy**: % of correct predictions
- **F1-Score**: Precision/recall balance
- **Confusion Matrix**: Visualizes errors

### Aggregated
- **Average Accuracy**: Mean accuracy across all horizons
- Used for early stopping and best model saving

---

## Generated Files

| File | Description |
|------|-------------|
| `v2/best_model.pt` | Best model (checkpoint) |
| `v2/training_curves.png` | Loss and Accuracy by horizon |
| `v2/confusion_matrices.png` | 4 confusion matrices (one per horizon) |

---

## Usage

### Training
```bash
# Prerequisite: run v1 first to generate the cache
python v2/model-v2.py
```

### Estimated Time
- **Loading + cache**: ~2 min
- **Training (15 epochs)**: ~8-10h on RTX 3080
- **Total**: ~10h

---

## Interpreting Results

### By Horizon

| Horizon | Expected (random) | Good | Excellent |
|---------|-------------------|------|-----------|
| D+1 | 33% | >40% | >50% |
| D+2 | 33% | >40% | >48% |
| D+5 | 33% | >38% | >45% |
| D+10 | 33% | >36% | >42% |

**Note:** Distant horizons (D+5, D+10) are harder as the signal weakens over time.

### Research Questions

1. **Which horizon is best predicted?** 
   - If D+1 > D+10: Risk Factors effect is short-term
   - If D+10 > D+1: Persistent effect

2. **Do the same patterns predict all horizons?**
   - Compare important features by horizon

---

## Limitations

1. **Fixed threshold (±0.5%)**: May not suit all markets
2. **No market context**: Ignores macroeconomic conditions
3. **Stationarity assumption**: Assumes 2019-2024 patterns persist

## Future Improvements

1. **Adaptive thresholds** by market volatility
2. **Additional features**: VIX, credit spreads, rating
3. **Temporal attention**: Compare current vs previous text
