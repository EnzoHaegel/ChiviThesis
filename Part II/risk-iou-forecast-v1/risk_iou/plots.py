"""Matplotlib figures for the deliverables (no display backend required)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_prf_by_threshold(by_threshold: dict, path: str) -> None:
    thresholds = sorted(by_threshold)
    prec = [by_threshold[t]["precision"] for t in thresholds]
    rec = [by_threshold[t]["recall"] for t in thresholds]
    f1 = [by_threshold[t]["f1"] for t in thresholds]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, prec, "o-", label="Precision")
    ax.plot(thresholds, rec, "s-", label="Recall")
    ax.plot(thresholds, f1, "^-", label="F1")
    ax.set_xlabel("IoU match threshold")
    ax.set_ylabel("Score")
    ax.set_title("Detection quality vs IoU threshold (test set)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_status_breakdown(summary: dict, path: str) -> None:
    labels = ["Persistent", "Intermittent", "Novel"]
    values = [summary["n_persistent"], summary["n_intermittent"], summary["n_novel"]]
    colors = ["#2c7fb8", "#7fcdbb", "#d95f0e"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("# risk terms in 2025")
    ax.set_title("2025 risk vocabulary: persistent vs novel")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_top_terms(df: pd.DataFrame, value_col: str, title: str, path: str, top: int = 15) -> None:
    if df.empty:
        # still emit a placeholder so the deliverable list is complete
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no terms", ha="center", va="center")
        ax.set_title(title)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(d))))
    ax.barh(d["term"], d[value_col], color="#2c7fb8")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_category_distribution(term_table: pd.DataFrame, path: str) -> None:
    if term_table.empty:
        return
    piv = term_table.groupby(["category", "status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    piv.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("# terms")
    ax.set_title("2025 risk terms by category and status")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
