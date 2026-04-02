"""Base evaluator class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
 
 
class BaseEvaluator(ABC):
    """Base class for all evaluators.
 
    All evaluators should inherit from this class and implement
    the required abstract methods.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize evaluator.
 
        Args:
            config: Evaluator configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self.metrics = {}
        self.results = []
 
    @abstractmethod
    def evaluate(self, predictions, targets) -> Dict[str, float]:
        """Evaluate predictions against targets.
 
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
 
        Returns:
            Dictionary of metric names to values.
        """
        pass
 
    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics.
 
        Returns:
            Dictionary of aggregated metric values.
        """
        pass
 
    def reset(self) -> None:
        """Reset evaluator state."""
        self.metrics = {}
        self.results = []
 
    def update(self, predictions, targets) -> None:
        """Update evaluator with new results.
 
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        """
        result = self.evaluate(predictions, targets)
        self.results.append(result)
 
        # Update metrics
        for key, value in result.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
 
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics.
 
        Returns:
            Dictionary of metric names to current values.
        """
        return self.compute_metrics()
 
    def get_config(self) -> Dict[str, Any]:
        """Get evaluator configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()