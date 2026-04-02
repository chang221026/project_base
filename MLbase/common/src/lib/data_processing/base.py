"""Base data processor class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
 
 
class BaseDataProcessor(ABC):
    """Base class for all data processors.
 
    Data processors are responsible for cleaning and transforming raw data.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize data processor.
 
        Args:
            config: Processor configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self._fitted = False
        self._state = {}
 
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data.
 
        Args:
            data: Raw data to process.
 
        Returns:
            Processed data.
        """
        pass
 
    @abstractmethod
    def fit(self, data: Any) -> 'BaseDataProcessor':
        """Fit processor to data.
 
        Args:
            data: Data to fit on.
 
        Returns:
            Self for chaining.
        """
        self._fitted = True
        return self
 
    def fit_process(self, data: Any) -> Any:
        """Fit and process data in one step.
 
        Args:
            data: Data to fit and process.
 
        Returns:
            Processed data.
        """
        return self.fit(data).process(data)
 
    def inverse_process(self, data: Any) -> Any:
        """Inverse process data (if applicable).
 
        Args:
            data: Processed data.
 
        Returns:
            Original data format.
        """
        return data
 
    def get_state(self) -> Dict[str, Any]:
        """Get processor state.
 
        Returns:
            State dictionary.
        """
        return self._state.copy()
 
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set processor state.
 
        Args:
            state: State dictionary.
        """
        self._state = state.copy()
        self._fitted = True
 
    def is_fitted(self) -> bool:
        """Check if processor is fitted.
 
        Returns:
            True if fitted, False otherwise.
        """
        return self._fitted
 
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def __call__(self, data: Any) -> Any:
        """Call processor (alias for process)."""
        return self.process(data)