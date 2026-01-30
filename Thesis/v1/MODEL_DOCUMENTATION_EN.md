# Bond Price Prediction Model - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Technical Specifications](#technical-specifications)
5. [Training](#training)
6. [Metrics and Evaluation](#metrics-and-evaluation)

---

## Overview

### Objective
This model predicts **bond returns** from the **Risk Factors** text contained in SEC 10-Q reports. The goal is to find correlations between the language used in financial documents and bond price movements.

### Research Hypothesis
> The words and expressions used in the "Risk Factors" section of quarterly reports contain predictive signals about the future performance of the company's bonds.

### Data Used

| Source | Description | Volume |
|--------|-------------|--------|
| CSV (csv_raw/) | TRACE data with bond returns | 132,032 observations |
| SEC Filings | Risk Factors text (10-Q) | 1,035 tickers |
| Period | 2019-2024 | 6 years |

---

## Model Architecture

### Global View

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT                                        │
│            Risk Factors Text (max 384 tokens)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FinBERT ENCODER                                 │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Tokenizer: WordPiece (30,522 tokens)                       │    │
│  │  Input: [CLS] token1 token2 ... tokenN [SEP] [PAD] ...      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  BERT Embeddings (768 dimensions)                           │    │
│  │  = Token Emb + Position Emb + Segment Emb                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  12 Transformer Layers (Frozen: 1-11, Trainable: 12)        │    │
│  │  Each layer: Self-Attention → Add&Norm → FFN → Add&Norm     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Output: Hidden States [batch, 384, 768]                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ATTENTION POOLING                                 │
│                                                                      │
│    hidden_states [batch, 384, 768]                                   │
│           │                                                          │
│           ▼                                                          │
│    ┌──────────────┐                                                  │
│    │ Linear(768→192)                                                 │
│    │ Tanh()                                                          │
│    │ Linear(192→1)  │ → attention_weights [batch, 384]              │
│    │ Softmax(dim=1) │                                                │
│    └──────────────┘                                                  │
│           │                                                          │
│           ▼                                                          │
│    weighted_sum = Σ(hidden_states × attention_weights)               │
│                                                                      │
│  Output: pooled [batch, 768], attention_weights [batch, 384]         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                                │
│                                                                      │
│    Concatenate: [pooled (768) + text_length_norm (1)] = 769          │
│           │                                                          │
│           ▼                                                          │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │ Linear(769 → 256)                                         │     │
│    │ LayerNorm(256)                                            │     │
│    │ GELU()                                                    │     │
│    │ Dropout(0.3)                                              │     │
│    │ Linear(256 → 128)                                         │     │
│    │ LayerNorm(128)                                            │     │
│    │ GELU()                                                    │     │
│    │ Dropout(0.3)                                              │     │
│    └──────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Output: features [batch, 128]                                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MULTI-TASK REGRESSION HEADS                        │
│                                                                      │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│    │ Head 1  │  │ Head 2  │  │ Head 3  │  │ Head 4  │  │ Head 5  │  │
│    │Linear   │  │Linear   │  │Linear   │  │Linear   │  │Linear   │  │
│    │(128→1)  │  │(128→1)  │  │(128→1)  │  │(128→1)  │  │(128→1)  │  │
│    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │
│         │           │           │           │           │           │
│         ▼           ▼           ▼           ▼           ▼           │
│      T+0         T+1         T+2         T+5         T+10           │
│   bond_return  return+1    return+2    return+5    return+10        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Step 1: Loading CSV Files

```python
csv_raw/
├── Earning_announcement_with_TRACE_2019_*.csv  (19,405 rows)
├── Earning_announcement_with_TRACE_2020_*.csv  (19,145 rows)
├── Earning_announcement_with_TRACE_2021_*.csv  (20,421 rows)
├── Earning_announcement_with_TRACE_2022_*.csv  (21,882 rows)
├── Earning_announcement_with_TRACE_2023_*.csv  (23,911 rows)
└── Earning_announcement_with_TRACE_2024_*.csv  (27,268 rows)
                                        Total: 132,032 rows
```

**Key columns used:**
- `tic`: Stock ticker
- `rdq`: Report Date Quarterly
- `bond_return`: Day 0 return
- `bond_return_plus_day_X`: Return at D+X

### Step 2: Scanning SEC Filings

```python
sec_10q/sec-edgar-filings/
├── AAPL/
│   └── 10-Q/
│       ├── 0001193125-19-023456/
│       │   └── full-submission_rf.txt  ← First line = date (YYYYMMDD)
│       └── ...
├── MSFT/
│   └── ...
└── ... (1,053 tickers)
```

**Format of `full-submission_rf.txt`:**
```
20230315                          ← Filing date (line 1)
Risks, Uncertainties and Other    ← Risk Factors text (lines 2+)
Factors That May Affect Future
Results Our operating results...
```

### Step 3: Data Matching

For each CSV observation:
1. Extract ticker and `rdq` date
2. Find the most recent SEC filing **before** `rdq`
3. Associate Risk Factors text with bond returns

```
CSV Observation:              Corresponding SEC Filing:
├── ticker: AAPL              ├── Date: 2023-02-15
├── rdq: 2023-03-20   →       └── Text: "Our business is subject
└── bond_return: 0.012            to risks including..."
```

**Result:** 95,315 matched observations (72% of data)

---

## Technical Specifications

### FinBERT

FinBERT is a version of BERT pre-trained on financial texts.

| Characteristic | Value |
|----------------|-------|
| Base model | BERT-base |
| Vocabulary | 30,522 tokens |
| Dimensions | 768 |
| Layers | 12 Transformer layers |
| Attention heads | 12 |
| Total parameters | ~110M |
| Pre-training | Financial texts (Reuters, SEC) |

**Why FinBERT instead of vanilla BERT?**
- Vocabulary adapted to financial domain
- Understands jargon: "liquidity risk", "covenant breach", "EBITDA"
- Better representation of financial sentiment

### Attention Pooling

Instead of simply using the `[CLS]` token, we learn **attention weights** for each token:

```python
# Formula
attention_weights = Softmax(Linear(Tanh(Linear(hidden_states))))
pooled = Σ(hidden_states × attention_weights)
```

**Advantages:**
1. Automatically identifies important words
2. Enables interpretability (which words influence predictions)
3. Better aggregation than simple [CLS]

### Multi-Task Learning

The model predicts **5 time horizons simultaneously**:

| Target | Description | Meaning |
|--------|-------------|---------|
| `bond_return` | Return at T+0 | Immediate reaction |
| `bond_return_plus_day_1` | Return at T+1 | Next day reaction |
| `bond_return_plus_day_2` | Return at T+2 | Short term |
| `bond_return_plus_day_5` | Return at T+5 | Medium term |
| `bond_return_plus_day_10` | Return at T+10 | Long term |

**Multi-task advantages:**
- Implicit regularization (shared representations)
- Captures different temporal dynamics
- Improves generalization

### Gradient Checkpointing

To save GPU memory, we use **gradient checkpointing**:

```python
self.encoder.gradient_checkpointing_enable()
```

**Principle:** Instead of storing all intermediate activations, we recompute them during the backward pass.

| Without Checkpointing | With Checkpointing |
|-----------------------|-------------------|
| VRAM: ~8GB | VRAM: ~4GB |
| Speed: 100% | Speed: ~80% |

---

## Training

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch size | 4 | GPU memory limit |
| Gradient accumulation | 4 | Effective batch = 16 |
| Learning rate | 2e-5 | Standard for fine-tuning |
| Max sequence length | 384 | Memory/context tradeoff |
| Epochs | 20 | With early stopping |
| Dropout | 0.3 | Strong regularization |
| Early stopping patience | 5 | Prevents overfitting |

### Loss Function

**Masked MSE Loss** - We compute MSE only on valid targets:

```python
def masked_mse_loss(predictions, targets, mask):
    diff = (predictions - targets) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum()
```

Why mask? Some observations don't have all returns available (missing data).

### Optimization

```python
# AdamW with weight decay
optimizer = AdamW(params, lr=2e-5, weight_decay=0.01)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

# Mixed precision (FP16)
scaler = GradScaler('cuda')
with autocast('cuda'):
    predictions, _ = model(input_ids, attention_mask)
```

**Mixed Precision (FP16):**
- Reduces memory usage by ~50%
- Accelerates GPU Tensor Core computations
- Precision maintained through GradScaler

---

## Metrics and Evaluation

### Computed Metrics

1. **MSE (Mean Squared Error)**: Prediction error
2. **Direction Accuracy**: % of times we predict the correct sign (+/-)

### Interpretability

After training, we extract the **most important words**:

```python
important_words = extract_important_words(model, tokenizer, text, device)
```

**Example output:**
```
Top 20 Most Important Words:
   1. litigation         (weight: 0.0342)
   2. regulatory         (weight: 0.0298)
   3. volatility         (weight: 0.0276)
   4. uncertainty        (weight: 0.0254)
   5. default            (weight: 0.0231)
   ...
```

These words have the most influence on bond return predictions.

---

## Generated Files

| File | Description |
|------|-------------|
| `v1/best_model.pt` | Best model weights |
| `v1/training_curves.png` | Train/val loss curves |
| `v1/filings_cache.json` | SEC filings cache |

---

## Limitations and Future Improvements

### Limitations
1. **Text length**: Truncated to 384 tokens (loses ~60% of text)
2. **Temporal matching**: Single filing per observation
3. **Control variables**: No market features

### Possible Improvements
1. **Longformer**: Handle 4096 tokens
2. **Market-adjusted returns**: Subtract market return
3. **Ensemble**: Combine with numerical features
4. **Sentiment scores**: Add FinBERT sentiment as feature

---

## References

- [FinBERT: Financial Sentiment Analysis](https://huggingface.co/yiyanghkust/finbert-tone)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
