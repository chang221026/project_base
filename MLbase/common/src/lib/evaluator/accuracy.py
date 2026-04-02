"""Accuracy evaluator implementation."""
from typing import Any, Dict, Optional
 
import numpy as np
import torch
 
from lib.evaluator import EVALUATORS, BaseEvaluator
 
 
@EVALUATORS.register('AccuracyEvaluator')
class AccuracyEvaluator(BaseEvaluator):
    """Accuracy evaluator for classification tasks.
 
    Computes accuracy and optionally additional metrics.
 
    Config parameters:
        top_k: Top-k accuracy (default: 1 for standard accuracy).
        num_classes: Number of classes (optional, for per-class accuracy).
        average: Averaging method for multi-class metrics ('micro', 'macro', 'weighted').
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 1,
        num_classes: Optional[int] = None,
        average: str = 'macro',
        **kwargs
    ):
        """Initialize AccuracyEvaluator.
 
        Args:
            config: Evaluator configuration dictionary.
            top_k: Top-k accuracy (1 for standard accuracy).
            num_classes: Number of classes (optional).
            average: Averaging method for multi-class metrics.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'top_k': top_k,
            'num_classes': num_classes,
            'average': average,
        })
        self.config.update(kwargs)
 
        self.top_k = self.config['top_k']
        self.num_classes = self.config['num_classes']
        self.average = self.config['average']
 
        # Tracking variables
        self._correct = 0
        self._total = 0
        self._class_correct = {}
        self._class_total = {}
 
    def evaluate(self, predictions, targets) -> Dict[str, float]:
        """Evaluate predictions against targets.
 
        Args:
            predictions: Model predictions (logits or probabilities).
                Shape: (batch_size, num_classes) or (batch_size,) for class labels.
            targets: Ground truth labels. Shape: (batch_size,).
 
        Returns:
            Dictionary of metric names to values.
        """
        # Convert to numpy if torch tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
 
        # Get predicted classes
        if predictions.ndim == 2:
            # Logits or probabilities
            if self.top_k == 1:
                predicted = np.argmax(predictions, axis=1)
            else:
                # Top-k accuracy
                top_k_preds = np.argsort(predictions, axis=1)[:, -self.top_k:]
                predicted = top_k_preds  # Will be handled differently below
        else:
            predicted = predictions
 
        # Compute accuracy
        if self.top_k == 1:
            correct = np.sum(predicted == targets)
            total = len(targets)
            accuracy = correct / total if total > 0 else 0.0
        else:
            # Top-k accuracy
            correct = 0
            total = len(targets)
            for i in range(total):
                if targets[i] in predicted[i]:
                    correct += 1
            accuracy = correct / total if total > 0 else 0.0
 
        metrics = {f'accuracy_top{self.top_k}': accuracy}
 
        # Track per-class accuracy if num_classes is set
        if self.num_classes is not None:
            for cls in range(self.num_classes):
                mask = targets == cls
                if np.any(mask):
                    cls_correct = np.sum((predicted[mask] == cls))
                    cls_total = np.sum(mask)
                    metrics[f'accuracy_class_{cls}'] = cls_correct / cls_total if cls_total > 0 else 0.0
 
        return metrics
 
    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics.
 
        Returns:
            Dictionary of aggregated metric values.
        """
        if not self.results:
            return {f'accuracy_top{self.top_k}': 0.0}
 
        # Average all results
        aggregated = {}
        for key in self.results[0].keys():
            values = [r[key] for r in self.results if key in r]
            aggregated[key] = np.mean(values) if values else 0.0
 
        return aggregated
 
    def reset(self) -> None:
        """Reset evaluator state."""
        super().reset()
        self._correct = 0
        self._total = 0
        self._class_correct = {}
        self._class_total = {}
 
    def __repr__(self) -> str:
        return f"AccuracyEvaluator(top_k={self.top_k}, num_classes={self.num_classes})"