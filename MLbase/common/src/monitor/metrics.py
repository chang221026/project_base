"""Post-training evaluation metrics module.

Provides comprehensive evaluation analysis for trained models, including
classification metrics (confusion matrix, ROC, PR, classification report),
regression metrics (MSE, MAE, R², MAPE), and a MetricCollection for
composing multiple metrics.

NOTE: This module is for **post-training deep analysis**, complementing
      `lib/evaluator/` which provides lightweight scalar metrics used
      during the training loop.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.logger import get_logger


logger = get_logger()


def _to_numpy(data) -> np.ndarray:
    """Convert input to numpy array.

    Supports torch.Tensor, list, tuple, and numpy array inputs.

    Args:
        data: Input data.

    Returns:
        Numpy array.
    """
    if isinstance(data, np.ndarray):
        return data
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(data)


# =============================================================================
# Evaluation Result Container
# =============================================================================

class EvaluationResult:
    """Container for evaluation results.

    Separates scalar metrics (loggable to ExperimentTracker) from
    structured data (confusion matrix, curve data, etc.).
    """

    def __init__(self):
        self.scalars: Dict[str, float] = {}
        self.data: Dict[str, Any] = {}

    def add_scalar(self, key: str, value: float) -> None:
        """Add a scalar metric.

        Args:
            key: Metric name.
            value: Metric value.
        """
        self.scalars[key] = value

    def add_data(self, key: str, value: Any) -> None:
        """Add structured data (non-scalar).

        Args:
            key: Data name.
            value: Data (array, dict, etc.).
        """
        self.data[key] = value

    def merge(self, other: 'EvaluationResult') -> None:
        """Merge another result into this one.

        Args:
            other: Result to merge.
        """
        self.scalars.update(other.scalars)
        self.data.update(other.data)

    def __repr__(self) -> str:
        scalar_str = ", ".join(f"{k}={v:.4f}" for k, v in self.scalars.items())
        data_keys = list(self.data.keys())
        return f"EvaluationResult(scalars=[{scalar_str}], data_keys={data_keys})"


# =============================================================================
# Base Metric
# =============================================================================

class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        """Compute the metric.

        Args:
            predictions: Model predictions.
            targets: Ground truth labels/values.

        Returns:
            EvaluationResult with computed metrics.
        """
        pass


# =============================================================================
# Classification Metrics
# =============================================================================

class ConfusionMatrixMetric(BaseMetric):
    """Computes confusion matrix for classification.

    Args:
        num_classes: Number of classes. Auto-detected if None.
        normalize: Normalization mode ('true', 'pred', 'all', or None).
    """

    def __init__(self, num_classes: Optional[int] = None,
                 normalize: Optional[str] = None):
        self.num_classes = num_classes
        self.normalize = normalize

    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        predictions = _to_numpy(predictions)
        targets = _to_numpy(targets)

        # Convert probabilities to class indices
        if predictions.ndim == 2:
            predictions = predictions.argmax(axis=1)

        targets = targets.astype(int).ravel()
        predictions = predictions.astype(int).ravel()

        n_classes = self.num_classes or max(targets.max(), predictions.max()) + 1
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)

        for t, p in zip(targets, predictions):
            cm[t, p] += 1

        result = EvaluationResult()
        result.add_data("confusion_matrix", cm)

        if self.normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
            result.add_data("confusion_matrix_normalized", cm_norm)
        elif self.normalize == "pred":
            col_sums = cm.sum(axis=0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.where(col_sums > 0, cm / col_sums, 0.0)
            result.add_data("confusion_matrix_normalized", cm_norm)
        elif self.normalize == "all":
            total = cm.sum()
            cm_norm = cm / total if total > 0 else cm.astype(float)
            result.add_data("confusion_matrix_normalized", cm_norm)

        # Derive per-class accuracy
        for i in range(n_classes):
            total_i = cm[i].sum()
            if total_i > 0:
                result.add_scalar(f"class_{i}_accuracy", cm[i, i] / total_i)

        return result


class ClassificationReport(BaseMetric):
    """Computes precision, recall, F1-score for classification.

    Args:
        average: Averaging strategy for multi-class ('macro', 'micro', 'weighted').
        num_classes: Number of classes. Auto-detected if None.
    """

    def __init__(self, average: str = "macro",
                 num_classes: Optional[int] = None):
        self.average = average
        self.num_classes = num_classes

    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        predictions = _to_numpy(predictions)
        targets = _to_numpy(targets)

        if predictions.ndim == 2:
            predictions = predictions.argmax(axis=1)

        targets = targets.astype(int).ravel()
        predictions = predictions.astype(int).ravel()

        n_classes = self.num_classes or max(targets.max(), predictions.max()) + 1

        # Per-class TP, FP, FN
        tp = np.zeros(n_classes)
        fp = np.zeros(n_classes)
        fn = np.zeros(n_classes)

        for t, p in zip(targets, predictions):
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        # Per-class precision, recall, F1
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0.0
            )

        support = np.array([np.sum(targets == i) for i in range(n_classes)])

        result = EvaluationResult()

        # Per-class metrics
        per_class = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }
        result.add_data("classification_report", per_class)

        # Averaged metrics
        if self.average == "micro":
            total_tp = tp.sum()
            total_fp = fp.sum()
            total_fn = fn.sum()
            avg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            avg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0
        elif self.average == "weighted":
            total_support = support.sum()
            if total_support > 0:
                weights = support / total_support
                avg_p = float(np.sum(precision * weights))
                avg_r = float(np.sum(recall * weights))
                avg_f1 = float(np.sum(f1 * weights))
            else:
                avg_p = avg_r = avg_f1 = 0.0
        else:  # macro
            mask = support > 0
            avg_p = float(precision[mask].mean()) if mask.any() else 0.0
            avg_r = float(recall[mask].mean()) if mask.any() else 0.0
            avg_f1 = float(f1[mask].mean()) if mask.any() else 0.0

        result.add_scalar("precision", avg_p)
        result.add_scalar("recall", avg_r)
        result.add_scalar("f1_score", avg_f1)
        result.add_scalar("accuracy", float(tp.sum() / len(targets)) if len(targets) > 0 else 0.0)

        return result


class ROCCurveMetric(BaseMetric):
    """Computes ROC curve data and AUC score.

    For binary classification. For multi-class, uses one-vs-rest.

    Args:
        num_thresholds: Number of threshold points for the curve.
    """

    def __init__(self, num_thresholds: int = 200):
        self.num_thresholds = num_thresholds

    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        """Compute ROC curve.

        Args:
            predictions: Probability scores (N,) for binary or (N, C) for multi-class.
            targets: Ground truth labels.

        Returns:
            EvaluationResult with ROC curve data and AUC.
        """
        predictions = _to_numpy(predictions).astype(float)
        targets = _to_numpy(targets).astype(int).ravel()

        result = EvaluationResult()

        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 2):
            # Binary classification
            if predictions.ndim == 2:
                scores = predictions[:, 1]
            else:
                scores = predictions

            fpr, tpr, thresholds = self._binary_roc(scores, targets)
            auc = self._auc(fpr, tpr)

            result.add_data("roc_curve", {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            })
            result.add_scalar("auc_roc", auc)

        elif predictions.ndim == 2:
            # Multi-class: one-vs-rest
            n_classes = predictions.shape[1]
            per_class_auc = []

            for c in range(n_classes):
                binary_targets = (targets == c).astype(int)
                scores = predictions[:, c]

                if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
                    continue

                fpr, tpr, thresholds = self._binary_roc(scores, binary_targets)
                auc = self._auc(fpr, tpr)
                per_class_auc.append(auc)

                result.add_data(f"roc_curve_class_{c}", {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                })
                result.add_scalar(f"auc_roc_class_{c}", auc)

            if per_class_auc:
                result.add_scalar("auc_roc_macro", float(np.mean(per_class_auc)))

        return result

    def _binary_roc(self, scores: np.ndarray,
                    targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute binary ROC curve."""
        thresholds = np.linspace(1.0, 0.0, self.num_thresholds)
        fpr_list = []
        tpr_list = []

        total_pos = targets.sum()
        total_neg = len(targets) - total_pos

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()

            tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
            fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)

        return np.array(fpr_list), np.array(tpr_list), thresholds

    @staticmethod
    def _auc(x: np.ndarray, y: np.ndarray) -> float:
        """Compute area under curve using trapezoidal rule."""
        # Sort by x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        return float(np.trapz(y_sorted, x_sorted))


class PRCurveMetric(BaseMetric):
    """Computes Precision-Recall curve data and average precision.

    Args:
        num_thresholds: Number of threshold points for the curve.
    """

    def __init__(self, num_thresholds: int = 200):
        self.num_thresholds = num_thresholds

    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        """Compute PR curve.

        Args:
            predictions: Probability scores (N,) for binary or (N, C) for multi-class.
            targets: Ground truth labels.

        Returns:
            EvaluationResult with PR curve data and average precision.
        """
        predictions = _to_numpy(predictions).astype(float)
        targets = _to_numpy(targets).astype(int).ravel()

        result = EvaluationResult()

        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 2):
            scores = predictions[:, 1] if predictions.ndim == 2 else predictions

            precision, recall, thresholds = self._binary_pr(scores, targets)
            ap = self._average_precision(precision, recall)

            result.add_data("pr_curve", {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
            })
            result.add_scalar("average_precision", ap)

        elif predictions.ndim == 2:
            n_classes = predictions.shape[1]
            per_class_ap = []

            for c in range(n_classes):
                binary_targets = (targets == c).astype(int)
                scores = predictions[:, c]

                if binary_targets.sum() == 0:
                    continue

                precision, recall, thresholds = self._binary_pr(scores, binary_targets)
                ap = self._average_precision(precision, recall)
                per_class_ap.append(ap)

                result.add_data(f"pr_curve_class_{c}", {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                })
                result.add_scalar(f"average_precision_class_{c}", ap)

            if per_class_ap:
                result.add_scalar("mean_average_precision", float(np.mean(per_class_ap)))

        return result

    def _binary_pr(self, scores: np.ndarray,
                   targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute binary PR curve."""
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)
        precision_list = []
        recall_list = []
        total_pos = targets.sum()

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / total_pos if total_pos > 0 else 0.0

            precision_list.append(prec)
            recall_list.append(rec)

        return np.array(precision_list), np.array(recall_list), thresholds

    @staticmethod
    def _average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute average precision (area under PR curve)."""
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        return float(np.trapz(precision_sorted, recall_sorted))


# =============================================================================
# Regression Metrics
# =============================================================================

class RegressionMetrics(BaseMetric):
    """Computes comprehensive regression metrics.

    Computes MSE, RMSE, MAE, R², and MAPE.
    """

    def compute(self, predictions: np.ndarray,
                targets: np.ndarray) -> EvaluationResult:
        predictions = _to_numpy(predictions).ravel().astype(float)
        targets = _to_numpy(targets).ravel().astype(float)

        result = EvaluationResult()

        residuals = targets - predictions

        # MSE
        mse = float(np.mean(residuals ** 2))
        result.add_scalar("mse", mse)

        # RMSE
        result.add_scalar("rmse", float(np.sqrt(mse)))

        # MAE
        result.add_scalar("mae", float(np.mean(np.abs(residuals))))

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        result.add_scalar("r2", float(r2))

        # MAPE (skip zero targets)
        non_zero = np.abs(targets) > 1e-10
        if non_zero.any():
            mape = float(np.mean(np.abs(residuals[non_zero] / targets[non_zero])) * 100)
            result.add_scalar("mape", mape)

        # Residuals for diagnostics
        result.add_data("residuals", residuals)

        return result


# =============================================================================
# MetricCollection
# =============================================================================

class MetricCollection:
    """Compose multiple metrics and compute them in one call.

    Example:
        metrics = MetricCollection([
            ClassificationReport(average='macro'),
            ConfusionMatrixMetric(),
            ROCCurveMetric(),
        ])
        result = metrics.compute(predictions, targets)
        # result.scalars -> {'precision': ..., 'recall': ..., 'f1_score': ..., ...}
        # result.data -> {'confusion_matrix': ..., 'roc_curve': ..., ...}
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        """Initialize collection.

        Args:
            metrics: List of BaseMetric instances.
        """
        self.metrics: List[BaseMetric] = metrics or []

    def add(self, metric: BaseMetric) -> 'MetricCollection':
        """Add a metric to the collection.

        Args:
            metric: Metric instance.

        Returns:
            Self for chaining.
        """
        self.metrics.append(metric)
        return self

    def compute(self, predictions, targets) -> EvaluationResult:
        """Compute all metrics.

        Args:
            predictions: Model predictions (numpy array or torch tensor).
            targets: Ground truth (numpy array or torch tensor).

        Returns:
            Merged EvaluationResult from all metrics.
        """
        predictions = _to_numpy(predictions)
        targets = _to_numpy(targets)

        combined = EvaluationResult()
        for metric in self.metrics:
            result = metric.compute(predictions, targets)
            combined.merge(result)

        return combined


# =============================================================================
# Convenience factory functions
# =============================================================================

def classification_metrics(average: str = "macro",
                           num_classes: Optional[int] = None,
                           include_roc: bool = True,
                           include_pr: bool = True) -> MetricCollection:
    """Create a standard classification metric collection.

    Args:
        average: Averaging strategy for multi-class.
        num_classes: Number of classes.
        include_roc: Include ROC curve and AUC.
        include_pr: Include PR curve and average precision.

    Returns:
        MetricCollection for classification evaluation.
    """
    metrics = MetricCollection([
        ClassificationReport(average=average, num_classes=num_classes),
        ConfusionMatrixMetric(num_classes=num_classes),
    ])
    if include_roc:
        metrics.add(ROCCurveMetric())
    if include_pr:
        metrics.add(PRCurveMetric())
    return metrics


def regression_metrics() -> MetricCollection:
    """Create a standard regression metric collection.

    Returns:
        MetricCollection for regression evaluation.
    """
    return MetricCollection([RegressionMetrics()])
