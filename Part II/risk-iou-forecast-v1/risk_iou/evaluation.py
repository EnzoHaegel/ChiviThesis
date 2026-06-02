"""Phase 3 — second-stage IoU evaluation with NMS and F1.

Detection pipeline per document:
    score candidates (model)  ->  threshold  ->  NMS  ->  IoU-match to ground truth

A detection is a true positive (TP) if it can be greedily matched (highest score
first) to a still-unmatched ground-truth span with IoU >= the match threshold.
Unmatched detections are false positives (FP); unmatched ground-truth spans are
false negatives (FN). Precision, recall and F1 follow from TP/FP/FN exactly as in
object detection. We sweep several IoU thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .labeling import DocAnnotation
from .model import RiskDetectorModel
from .nms import nms
from .spans import Span, iou


@dataclass
class Detection:
    span: Span
    filing_id: str


def detect_document(
    ann: DocAnnotation,
    model: RiskDetectorModel,
    cfg: Config,
) -> list[Span]:
    """Score candidates, drop sub-threshold ones, then apply NMS."""
    scored = model.score_spans(ann.candidates)
    kept = [s for s in scored if s.score >= cfg.score_threshold]
    return nms(kept, iou_threshold=cfg.nms_iou_threshold)


def match(
    detections: list[Span], gt: list[Span], iou_threshold: float
) -> tuple[int, int, int]:
    """Greedy score-ordered matching → (tp, fp, fn)."""
    if not detections and not gt:
        return 0, 0, 0
    dets = sorted(detections, key=lambda s: -s.score)
    matched_gt = [False] * len(gt)
    tp = 0
    fp = 0
    for d in dets:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt):
            if matched_gt[j]:
                continue
            v = iou(d, g)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched_gt[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = matched_gt.count(False)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


@dataclass
class EvaluationResult:
    by_threshold: dict[float, dict]
    candidate_accuracy: dict
    n_documents: int
    n_detections: int
    n_ground_truth: int
    detections_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _candidate_level_accuracy(
    annotations: list[DocAnnotation], model: RiskDetectorModel, cfg: Config
) -> dict:
    """Window-level accuracy/precision/recall: model decision vs IoU-derived label."""
    tp = tn = fp = fn = 0
    for ann in annotations:
        if not ann.candidates:
            continue
        scored = model.score_spans(ann.candidates)
        for s, y in zip(scored, ann.labels):
            pred = 1 if s.score >= cfg.score_threshold else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
            else:
                tn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    out = _prf(tp, fp, fn)
    out["tn"] = tn
    out["accuracy"] = round(acc, 4)
    return out


def evaluate(
    annotations: list[DocAnnotation],
    model: RiskDetectorModel,
    cfg: Config,
    iou_thresholds: tuple[float, ...] | None = None,
    collect_detections: bool = True,
) -> EvaluationResult:
    thresholds = iou_thresholds or cfg.eval_iou_thresholds
    agg = {t: [0, 0, 0] for t in thresholds}   # t -> [tp, fp, fn]
    det_rows = []
    n_det = 0
    n_gt = 0
    for ann in annotations:
        detections = detect_document(ann, model, cfg)
        n_det += len(detections)
        n_gt += len(ann.gt)
        for t in thresholds:
            tp, fp, fn = match(detections, ann.gt, t)
            agg[t][0] += tp
            agg[t][1] += fp
            agg[t][2] += fn
        if collect_detections:
            for d in detections:
                det_rows.append(
                    {
                        "filing_id": ann.filing.filing_id,
                        "quarter": ann.filing.quarter,
                        "date": ann.filing.date,
                        "term": d.text,
                        "token_start": d.start,
                        "token_end": d.end,
                        "score": round(d.score, 4),
                    }
                )
    by_threshold = {t: _prf(*agg[t]) for t in thresholds}
    cand_acc = _candidate_level_accuracy(annotations, model, cfg)
    return EvaluationResult(
        by_threshold=by_threshold,
        candidate_accuracy=cand_acc,
        n_documents=len(annotations),
        n_detections=n_det,
        n_ground_truth=n_gt,
        detections_df=pd.DataFrame(det_rows),
    )
