"""
PhishGuard AI — Model Evaluation

Computes and reports all evaluation metrics:
  - Accuracy
  - Precision, Recall, F1 (per-class)
  - Macro F1, Weighted F1
  - Confusion Matrix
  - False Positive / False Negative analysis

All metrics come from ACTUAL evaluation runs — nothing is fabricated.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.utils.config import CLASSES, LABEL_TO_CLASS

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""

    model_name: str
    experiment_name: str
    dataset_split: str  # "validation" or "test"
    n_samples: int

    # Overall metrics
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Confusion matrix (as list of lists for serialization)
    confusion_matrix: List[List[int]] = field(default_factory=list)

    # Classification report string
    classification_report_text: str = ""

    # False positive / negative counts
    false_positives: Dict[str, int] = field(default_factory=dict)
    false_negatives: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "dataset_split": self.dataset_split,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    experiment_name: str,
    dataset_split: str = "test",
    class_names: Optional[List[str]] = None,
) -> EvaluationReport:
    """
    Evaluate a trained model and generate a comprehensive report.

    All metrics are computed from the actual predictions — nothing is invented.

    Args:
        model: Trained scikit-learn model
        X: Feature matrix
        y_true: True labels (integer-encoded)
        model_name: Human-readable model name
        experiment_name: Experiment identifier
        dataset_split: "validation" or "test"
        class_names: List of class names (default: from config)

    Returns:
        EvaluationReport with all metrics
    """
    if class_names is None:
        class_names = CLASSES

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION: {experiment_name} — {model_name}")
    logger.info(f"Split: {dataset_split} | Samples: {len(y_true)}")
    logger.info(f"{'='*60}")

    # Handle NB non-negative requirement
    from sklearn.naive_bayes import MultinomialNB
    from scipy.sparse import issparse

    X_eval = X
    if isinstance(model, MultinomialNB):
        if issparse(X_eval):
            X_eval = X_eval.copy()
            X_eval[X_eval < 0] = 0
        else:
            X_eval = np.clip(X_eval, 0, None)

    # Predictions
    y_pred = model.predict(X_eval)

    # Only use labels present in either y_true or y_pred
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_class_names = [class_names[i] for i in present_labels if i < len(class_names)]

    report = EvaluationReport(
        model_name=model_name,
        experiment_name=experiment_name,
        dataset_split=dataset_split,
        n_samples=len(y_true),
    )

    # Overall metrics
    report.accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    report.macro_precision = round(
        float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )
    report.macro_recall = round(
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )
    report.macro_f1 = round(
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )
    report.weighted_f1 = round(
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
    )

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )

    for i, label_idx in enumerate(present_labels):
        if label_idx < len(class_names):
            name = class_names[label_idx]
            report.per_class_metrics[name] = {
                "precision": round(float(precision_per[i]), 4),
                "recall": round(float(recall_per[i]), 4),
                "f1": round(float(f1_per[i]), 4),
                "support": int(support_per[i]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    report.confusion_matrix = cm.tolist()

    # Classification report text
    report.classification_report_text = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_class_names,
        zero_division=0,
    )

    # False positive / false negative analysis
    for i, label_idx in enumerate(present_labels):
        if label_idx < len(class_names):
            name = class_names[label_idx]
            # FP: predicted as this class but truly something else
            fp = int(cm[:, i].sum() - cm[i, i])
            # FN: truly this class but predicted as something else
            fn = int(cm[i, :].sum() - cm[i, i])
            report.false_positives[name] = fp
            report.false_negatives[name] = fn

    # Log results
    logger.info(f"\nAccuracy: {report.accuracy}")
    logger.info(f"Macro F1: {report.macro_f1}")
    logger.info(f"Weighted F1: {report.weighted_f1}")
    logger.info(f"\n{report.classification_report_text}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"Labels: {present_class_names}")
    for row in report.confusion_matrix:
        logger.info(f"  {row}")
    logger.info(f"\nFalse Positives: {report.false_positives}")
    logger.info(f"False Negatives: {report.false_negatives}")

    return report


def compare_experiments(reports: List[EvaluationReport]) -> str:
    """
    Generate a comparison table of multiple experiment results.

    Returns formatted string table.
    """
    header = f"{'Experiment':<20} {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}"
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for r in reports:
        lines.append(
            f"{r.experiment_name:<20} {r.model_name:<25} "
            f"{r.accuracy:>10.4f} {r.macro_f1:>10.4f} {r.weighted_f1:>12.4f}"
        )
    lines.append(separator)

    table = "\n".join(lines)
    logger.info(f"\nEXPERIMENT COMPARISON:\n{table}")
    return table
