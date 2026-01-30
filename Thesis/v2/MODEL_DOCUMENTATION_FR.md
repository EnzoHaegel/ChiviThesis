# Modèle de Prédiction de Direction des Obligations V2
## Classification Multi-Horizon (J+1 à J+10)

## Vue d'ensemble

### Objectif
Ce modèle prédit la **direction future** (HAUSSE/BAISSE/STABLE) des retours obligataires à partir du texte des Risk Factors, pour **plusieurs horizons temporels après l'annonce de résultats**.

### Pourquoi Multi-Horizon ?

```
                     Annonce des résultats (rdq)
                              │
     Risk Factors             │
    publiés AVANT  ◄──────────┤
                              │
                              ▼
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ J-1 │ J+0 │ J+1 │ J+2 │ J+3 │ J+4 │ J+5 │ J+6 │ J+7 │ J+8 │ J+10│
    └─────┴─────┴──┬──┴──┬──┴─────┴─────┴──┬──┴─────┴─────┴─────┴──┬──┘
                   │     │                 │                       │
                   ▼     ▼                 ▼                       ▼
                 J+1   J+2               J+5                     J+10
                   │     │                 │                       │
                   └─────┴────────┬────────┴───────────────────────┘
                                  │
                            PRÉDICTIONS
                        (HAUSSE/BAISSE/STABLE)
```

**T+0 = jour de l'annonce** → Pas vraiment une prédiction, c'est simultané !  
**J+1 à J+10** → Ce sont de vraies **prédictions du futur** ✅

---

## Horizons Prédits

| Horizon | Description | Intérêt |
|---------|-------------|---------|
| **J+1** | Retour 1 jour après annonce | Réaction immédiate du marché |
| **J+2** | Retour 2 jours après | Confirmation de la tendance |
| **J+5** | Retour 5 jours après | Effet moyen terme (1 semaine) |
| **J+10** | Retour 10 jours après | Effet long terme (2 semaines) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ENTRÉE                                     │
│              Texte Risk Factors (max 256 tokens)                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FinBERT ENCODER                                 │
│  12 couches Transformer (Gelées: 1-11, Entraînée: 12)               │
│  Sortie: [CLS] token [batch, 768]                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COUCHE PARTAGÉE                                 │
│  Linear(768 → 256) → LayerNorm → GELU → Dropout(0.2)                │
│  Sortie: features partagées [batch, 256]                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   HEAD J+1        │ │   HEAD J+2        │ │   HEAD J+5        │ ...
│ Linear(256→64)    │ │ Linear(256→64)    │ │ Linear(256→64)    │
│ GELU → Dropout    │ │ GELU → Dropout    │ │ GELU → Dropout    │
│ Linear(64→3)      │ │ Linear(64→3)      │ │ Linear(64→3)      │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
          │                     │                     │
          ▼                     ▼                     ▼
    [P(↓), P(→), P(↑)]   [P(↓), P(→), P(↑)]   [P(↓), P(→), P(↑)]
       pour J+1              pour J+2              pour J+5
```

### Multi-Task Learning

Le modèle partage les représentations FinBERT et la couche cachée entre tous les horizons, puis utilise des **têtes spécialisées** pour chaque horizon.

**Avantages :**
- Régularisation implicite (partage de représentations)
- Plus efficace en mémoire qu'entraîner 4 modèles séparés
- Capture les patterns communs à tous les horizons

---

## Classification des Retours

### Seuils de Direction

```
        BAISSE            STABLE              HAUSSE
    ◄──────────────►◄─────────────────►◄──────────────►
         < -0.5%      -0.5% à +0.5%        > +0.5%
```

| Classe | Condition | Code |
|--------|-----------|------|
| BAISSE (DOWN) | return < -0.5% | 0 |
| STABLE (NEUTRAL) | -0.5% ≤ return ≤ +0.5% | 1 |
| HAUSSE (UP) | return > +0.5% | 2 |

---

## Gestion des Données Manquantes

Certaines observations n'ont pas tous les horizons disponibles. Le modèle utilise un **masque de loss** :

```python
# Pour chaque observation
labels = [class_J1, class_J2, class_J5, class_J10]
masks = [1, 1, 0, 1]  # 0 = donnée manquante

# Loss calculée uniquement sur les horizons valides
loss = masked_cross_entropy(logits, labels, masks)
```

---

## Métriques d'Évaluation

### Par Horizon
- **Accuracy** : % de prédictions correctes
- **F1-Score** : Équilibre précision/rappel
- **Matrice de confusion** : Visualise les erreurs

### Agrégée
- **Average Accuracy** : Moyenne des accuracies sur tous les horizons
- Utilisée pour l'early stopping et la sauvegarde du meilleur modèle

---

## Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `v2/best_model.pt` | Meilleur modèle (checkpoint) |
| `v2/training_curves.png` | Loss et Accuracy par horizon |
| `v2/confusion_matrices.png` | 4 matrices de confusion (une par horizon) |

---

## Utilisation

### Entraînement
```bash
# Prérequis: avoir exécuté v1 pour générer le cache
python v2/model-v2.py
```

### Temps Estimé
- **Chargement + cache**: ~2 min
- **Training (15 epochs)**: ~8-10h sur RTX 3080
- **Total**: ~10h

---

## Interprétation des Résultats

### Horizon par Horizon

| Horizon | Attendu (random) | Bon | Excellent |
|---------|------------------|-----|-----------|
| J+1 | 33% | >40% | >50% |
| J+2 | 33% | >40% | >48% |
| J+5 | 33% | >38% | >45% |
| J+10 | 33% | >36% | >42% |

**Note :** Les horizons lointains (J+5, J+10) sont plus difficiles car le signal s'affaiblit avec le temps.

### Questions de Recherche

1. **Quel horizon est le mieux prédit ?** 
   - Si J+1 > J+10 : L'effet des Risk Factors est court terme
   - Si J+10 > J+1 : Effet persistant

2. **Les mêmes patterns prédisent-ils tous les horizons ?**
   - Comparer les features importantes par horizon

---

## Limitations

1. **Seuil fixe (±0.5%)** : Peut ne pas convenir à tous les marchés
2. **Pas de contexte marché** : Ignore les conditions macroéconomiques
3. **Hypothèse de stationnarité** : Suppose que les patterns 2019-2024 persistent

## Améliorations Futures

1. **Seuils adaptatifs** par volatilité du marché
2. **Features additionnelles** : VIX, spreads de crédit, rating
3. **Temporal attention** : Comparer le texte actuel vs précédent
