"""Base loss function class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
 
 
class BaseLoss(ABC):
    """Base class for all loss functions.
 
    All loss functions should inherit from this class and implement
    the required abstract methods.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize loss function.
 
        Args:
            config: Loss configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
 
    @abstractmethod
    def compute(self, predictions, targets):
        """Compute loss.
 
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
 
        Returns:
            Computed loss value.
        """
        pass
 
    def __call__(self, predictions, targets):
        """Call loss function (alias for compute)."""
        return self.compute(predictions, targets)
 
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()