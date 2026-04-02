"""Base feature constructor class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
 
 
class BaseFeatureConstructor(ABC):
    """Base class for all feature constructors.
 
    Feature constructors are responsible for creating new features
    from existing data.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize feature constructor.
 
        Args:
            config: Constructor configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self._feature_names = []
        self._fitted = False
 
    @abstractmethod
    def construct(self, data: Any) -> Any:
        """Construct new features from data.
 
        Args:
            data: Input data.
 
        Returns:
            Data with new features.
        """
        pass
 
    @abstractmethod
    def fit(self, data: Any) -> 'BaseFeatureConstructor':
        """Fit constructor to data.
 
        Args:
            data: Data to fit on.
 
        Returns:
            Self for chaining.
        """
        self._fitted = True
        return self
 
    def fit_construct(self, data: Any) -> Any:
        """Fit and construct features in one step.
 
        Args:
            data: Data to fit and construct features from.
 
        Returns:
            Data with new features.
        """
        return self.fit(data).construct(data)
 
    def get_feature_names(self) -> List[str]:
        """Get names of constructed features.
 
        Returns:
            List of feature names.
        """
        return self._feature_names.copy()
 
    def get_config(self) -> Dict[str, Any]:
        """Get constructor configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def __call__(self, data: Any) -> Any:
        """Call constructor (alias for construct)."""
        return self.construct(data)