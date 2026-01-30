# Bond Price Prediction Model - Documentation Technique

## Table des Matières
1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du Modèle](#architecture-du-modèle)
3. [Pipeline de Données](#pipeline-de-données)
4. [Spécificités Techniques](#spécificités-techniques)
5. [Entraînement](#entraînement)
6. [Métriques et Évaluation](#métriques-et-évaluation)

---

## Vue d'ensemble

### Objectif
Ce modèle prédit les **retours obligataires** (bond returns) à partir du texte des **Risk Factors** contenus dans les rapports SEC 10-Q. L'idée est de trouver des corrélations entre le langage utilisé dans les documents financiers et les mouvements de prix des obligations.

### Hypothèse de Recherche
> Les mots et expressions utilisés dans la section "Risk Factors" des rapports trimestriels contiennent des signaux prédictifs sur la performance future des obligations de l'entreprise.

### Données Utilisées

| Source | Description | Volume |
|--------|-------------|--------|
| CSV (csv_raw/) | Données TRACE avec retours obligataires | 132,032 observations |
| SEC Filings | Texte des Risk Factors (10-Q) | 1,035 tickers |
| Période | 2019-2024 | 6 années |

---

## Architecture du Modèle

### Vue Globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT                                        │
│            Texte Risk Factors (max 384 tokens)                       │
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

## Pipeline de Données

### Étape 1: Chargement des CSV

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

**Colonnes clés utilisées:**
- `tic`: Ticker de l'action
- `rdq`: Date du rapport (Report Date Quarterly)
- `bond_return`: Retour du jour 0
- `bond_return_plus_day_X`: Retour à J+X

### Étape 2: Scan des Filings SEC

```python
sec_10q/sec-edgar-filings/
├── AAPL/
│   └── 10-Q/
│       ├── 0001193125-19-023456/
│       │   └── full-submission_rf.txt  ← Première ligne = date (YYYYMMDD)
│       └── ...
├── MSFT/
│   └── ...
└── ... (1,053 tickers)
```

**Format du fichier `full-submission_rf.txt`:**
```
20230315                          ← Date du filing (ligne 1)
Risks, Uncertainties and Other    ← Texte Risk Factors (lignes 2+)
Factors That May Affect Future
Results Our operating results...
```

### Étape 3: Matching des Données

Pour chaque observation dans les CSV:
1. Extraire le ticker et la date `rdq`
2. Trouver le filing SEC **le plus récent avant** `rdq`
3. Associer le texte Risk Factors aux retours obligataires

```
Observation CSV:          Filing SEC correspondant:
├── ticker: AAPL          ├── Date: 2023-02-15
├── rdq: 2023-03-20   →   └── Texte: "Our business is subject
└── bond_return: 0.012        to risks including..."
```

**Résultat:** 95,315 observations matchées (72% des données)

---

## Spécificités Techniques

### FinBERT

FinBERT est une version de BERT pré-entraînée sur des textes financiers.

| Caractéristique | Valeur |
|-----------------|--------|
| Modèle de base | BERT-base |
| Vocabulaire | 30,522 tokens |
| Dimensions | 768 |
| Couches | 12 Transformer layers |
| Têtes d'attention | 12 |
| Paramètres totaux | ~110M |
| Pré-entraînement | Textes financiers (Reuters, SEC) |

**Pourquoi FinBERT plutôt que BERT vanilla?**
- Vocabulaire adapté au domaine financier
- Comprend le jargon: "liquidity risk", "covenant breach", "EBITDA"
- Meilleure représentation du sentiment financier

### Attention Pooling

Au lieu d'utiliser simplement le token `[CLS]`, on apprend des **poids d'attention** pour chaque token:

```python
# Formule
attention_weights = Softmax(Linear(Tanh(Linear(hidden_states))))
pooled = Σ(hidden_states × attention_weights)
```

**Avantages:**
1. Identifie automatiquement les mots importants
2. Permet l'interprétabilité (quels mots influencent la prédiction)
3. Meilleure agrégation que le simple [CLS]

### Multi-Task Learning

Le modèle prédit **5 horizons temporels simultanément**:

| Target | Description | Signification |
|--------|-------------|---------------|
| `bond_return` | Retour à T+0 | Réaction immédiate |
| `bond_return_plus_day_1` | Retour à T+1 | Réaction lendemain |
| `bond_return_plus_day_2` | Retour à T+2 | Court terme |
| `bond_return_plus_day_5` | Retour à T+5 | Moyen terme |
| `bond_return_plus_day_10` | Retour à T+10 | Long terme |

**Avantages du multi-task:**
- Régularisation implicite (partage de représentations)
- Capture différentes dynamiques temporelles
- Améliore la généralisation

### Gradient Checkpointing

Pour économiser la mémoire GPU, on utilise le **gradient checkpointing**:

```python
self.encoder.gradient_checkpointing_enable()
```

**Principe:** Au lieu de stocker toutes les activations intermédiaires, on les recalcule pendant le backward pass.

| Sans Checkpointing | Avec Checkpointing |
|--------------------|-------------------|
| VRAM: ~8GB | VRAM: ~4GB |
| Vitesse: 100% | Vitesse: ~80% |

---

## Entraînement

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Batch size | 4 | Limite mémoire GPU |
| Gradient accumulation | 4 | Effective batch = 16 |
| Learning rate | 2e-5 | Standard pour fine-tuning |
| Max sequence length | 384 | Compromis mémoire/contexte |
| Epochs | 20 | Avec early stopping |
| Dropout | 0.3 | Régularisation forte |
| Early stopping patience | 5 | Évite overfitting |

### Loss Function

**Masked MSE Loss** - On calcule la MSE uniquement sur les targets valides:

```python
def masked_mse_loss(predictions, targets, mask):
    diff = (predictions - targets) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum()
```

Pourquoi masquer? Certaines observations n'ont pas tous les retours disponibles (données manquantes).

### Optimisation

```python
# AdamW avec weight decay
optimizer = AdamW(params, lr=2e-5, weight_decay=0.01)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

# Mixed precision (FP16)
scaler = GradScaler('cuda')
with autocast('cuda'):
    predictions, _ = model(input_ids, attention_mask)
```

**Mixed Precision (FP16):**
- Réduit utilisation mémoire de ~50%
- Accélère calculs sur GPU Tensor Cores
- Précision maintenue grâce au GradScaler

---

## Métriques et Évaluation

### Métriques Calculées

1. **MSE (Mean Squared Error)**: Erreur de prédiction
2. **Direction Accuracy**: % de fois où on prédit le bon signe (+/-)

### Interprétabilité

Après entraînement, on extrait les **mots les plus importants**:

```python
important_words = extract_important_words(model, tokenizer, text, device)
```

**Exemple de sortie:**
```
Top 20 Most Important Words:
   1. litigation         (weight: 0.0342)
   2. regulatory         (weight: 0.0298)
   3. volatility         (weight: 0.0276)
   4. uncertainty        (weight: 0.0254)
   5. default            (weight: 0.0231)
   ...
```

Ces mots sont ceux qui ont le plus d'influence sur les prédictions de retours obligataires.

---

## Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `v1/best_model.pt` | Poids du meilleur modèle |
| `v1/training_curves.png` | Courbes train/val loss |
| `v1/filings_cache.json` | Cache des filings SEC |

---

## Limitations et Améliorations Futures

### Limitations
1. **Longueur texte**: Tronqué à 384 tokens (perd ~60% du texte)
2. **Matching temporel**: Un seul filing par observation
3. **Variables de contrôle**: Pas de features de marché

### Améliorations Possibles
1. **Longformer**: Gérer 4096 tokens
2. **Market-adjusted returns**: Soustraire le retour du marché
3. **Ensemble**: Combiner avec features numériques
4. **Sentiment scores**: Ajouter FinBERT sentiment comme feature

---

## Références

- [FinBERT: Financial Sentiment Analysis](https://huggingface.co/yiyanghkust/finbert-tone)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
