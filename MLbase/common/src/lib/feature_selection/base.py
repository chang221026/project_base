"""Base feature selector class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
 
 
class BaseFeatureSelector(ABC):
    """Base class for all feature selectors.
 
    Feature selectors are responsible for selecting the most relevant
    features from the dataset.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize feature selector.
 
        Args:
            config: Selector configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self._selected_features = []
        self._feature_scores = {}
        self._fitted = False
 
    @abstractmethod
    def select(self, data: Any, target: Optional[Any] = None) -> Any:
        """Select features from data.
 
        Args:
            data: Input data.
            target: Optional target variable for supervised selection.
 
        Returns:
            Data with selected features.
        """
        pass
 
    @abstractmethod
    def fit(self, data: Any, target: Optional[Any] = None) -> 'BaseFeatureSelector':
        """Fit selector to data.
 
        Args:
            data: Data to fit on.
            target: Optional target variable.
 
        Returns:
            Self for chaining.
        """
        self._fitted = True
        return self
 
    def fit_select(self, data: Any, target: Optional[Any] = None) -> Any:
        """Fit and select features in one step.
 
        Args:
            data: Data to fit and select from.
            target: Optional target variable.
 
        Returns:
            Data with selected features.
        """
        return self.fit(data, target).select(data, target)
 
    def get_selected_features(self) -> List[str]:
        """Get names of selected features.
 
        Returns:
            List of selected feature names.
        """
        return self._selected_features.copy()
 
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores.
 
        Returns:
            Dictionary mapping feature names to scores.
        """
        return self._feature_scores.copy()
 
    def get_config(self) -> Dict[str, Any]:
        """Get selector configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def __call__(self, data: Any, target: Optional[Any] = None) -> Any:
        """Call selector (alias for select)."""
        return self.select(data, target)