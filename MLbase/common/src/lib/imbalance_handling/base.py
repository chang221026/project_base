"""Base imbalance handler class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
 
 
class BaseImbalanceHandler(ABC):
    """Base class for all imbalance handlers.
 
    Imbalance handlers are responsible for handling class imbalance
    in datasets through resampling or weighting techniques.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize imbalance handler.
 
        Args:
            config: Handler configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self._class_distribution = {}
        self._fitted = False
 
    @abstractmethod
    def handle(self, data: Any, target: Any) -> Tuple[Any, Any]:
        """Handle class imbalance in data.
 
        Args:
            data: Input data.
            target: Target labels.
 
        Returns:
            Tuple of (balanced_data, balanced_target).
        """
        pass
 
    @abstractmethod
    def fit(self, data: Any, target: Any) -> 'BaseImbalanceHandler':
        """Fit handler to data.
 
        Args:
            data: Data to fit on.
            target: Target labels.
 
        Returns:
            Self for chaining.
        """
        self._fitted = True
        return self
 
    def fit_handle(self, data: Any, target: Any) -> Tuple[Any, Any]:
        """Fit and handle imbalance in one step.
 
        Args:
            data: Data to fit and handle.
            target: Target labels.
 
        Returns:
            Tuple of (balanced_data, balanced_target).
        """
        return self.fit(data, target).handle(data, target)
 
    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for weighted loss.
 
        Returns:
            Dictionary mapping class labels to weights.
        """
        if not self._class_distribution:
            return {}
 
        total = sum(self._class_distribution.values())
        num_classes = len(self._class_distribution)
 
        weights = {}
        for cls, count in self._class_distribution.items():
            weights[cls] = total / (num_classes * count)
 
        return weights
 
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution.
 
        Returns:
            Dictionary mapping class labels to counts.
        """
        return self._class_distribution.copy()
 
    def get_config(self) -> Dict[str, Any]:
        """Get handler configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def __call__(self, data: Any, target: Any) -> Tuple[Any, Any]:
        """Call handler (alias for handle)."""
        return self.handle(data, target)
 
 
# Type alias for return type
Tuple = tuple