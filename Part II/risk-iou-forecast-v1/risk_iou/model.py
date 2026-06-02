"""Phase 2 — the risk-keyword detector model.

A candidate span is scored by a learned classifier (logistic regression over
character + word n-gram TF-IDF features of the span text plus its token length).
The learned probability replaces the lexicon's hard membership as the detection
confidence used downstream by NMS and IoU evaluation — i.e. the model "learns to
identify and classify risk keywords and patterns" rather than merely matching the
seed lexicon, which lets it generalise to unseen wordings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .spans import Span


def _build_pipeline(seed: int) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
                ),
                "span_text",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                ),
                "span_text",
            ),
            ("len", MaxAbsScaler(), ["span_len"]),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=4.0,
        random_state=seed,
    )
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray


def grouped_train_test_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> SplitResult:
    """80/20 split that keeps all candidates of one filing on the same side.

    Grouping by ``filing_id`` prevents train/test leakage of document-specific
    vocabulary, so the reported test metrics reflect generalisation to unseen
    filings rather than memorisation.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["filing_id"]))
    return SplitResult(train_idx=np.asarray(train_idx), test_idx=np.asarray(test_idx))


class RiskDetectorModel:
    """Thin wrapper around the sklearn pipeline with span-scoring helpers."""

    FEATURE_COLS = ["span_text", "span_len"]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pipeline = _build_pipeline(seed)
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "RiskDetectorModel":
        X = df[self.FEATURE_COLS]
        y = df["label"].to_numpy()
        self.pipeline.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.FEATURE_COLS]
        return self.pipeline.predict_proba(X)[:, 1]

    def score_spans(self, spans: list[Span]) -> list[Span]:
        """Assign each span a learned confidence into ``span.score`` (new objects)."""
        if not spans:
            return []
        frame = pd.DataFrame(
            {
                "span_text": [s.text for s in spans],
                "span_len": [s.length for s in spans],
            }
        )
        probs = self.predict_proba(frame)
        scored = []
        for s, p in zip(spans, probs):
            scored.append(
                Span(start=s.start, end=s.end, text=s.text, category=s.category, score=float(p))
            )
        return scored

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"pipeline": self.pipeline, "seed": self.seed, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: str) -> "RiskDetectorModel":
        import joblib

        blob = joblib.load(path)
        m = cls(seed=blob["seed"])
        m.pipeline = blob["pipeline"]
        m.fitted = blob["fitted"]
        return m
