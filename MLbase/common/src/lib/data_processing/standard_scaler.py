"""Standard scaler data processor implementation."""
from typing import Any, Dict, Optional
 
import numpy as np
 
from lib.data_processing import DATA_PROCESSORS, BaseDataProcessor
 
 
@DATA_PROCESSORS.register('StandardScaler')
class StandardScaler(BaseDataProcessor):
    """Standard scaler for Z-score normalization.
 
    Transforms features to have zero mean and unit variance.
 
    Config parameters:
        with_mean: Whether to center data (default: True).
        with_std: Whether to scale to unit variance (default: True).
        copy: Whether to create a copy of the data (default: True).
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        copy: bool = True,
        **kwargs
    ):
        """Initialize StandardScaler.
 
        Args:
            config: Processor configuration dictionary.
            with_mean: Whether to center data.
            with_std: Whether to scale to unit variance.
            copy: Whether to create a copy of the data.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'with_mean': with_mean,
            'with_std': with_std,
            'copy': copy,
        })
        self.config.update(kwargs)
 
        self.with_mean = self.config['with_mean']
        self.with_std = self.config['with_std']
        self.copy = self.config['copy']
 
        self._mean = None
        self._std = None
        self._state = {}
 
    def fit(self, data: Any) -> 'StandardScaler':
        """Fit scaler to data.
 
        Computes mean and standard deviation for each feature.
 
        Args:
            data: Data to fit on (numpy array).
 
        Returns:
            Self for chaining.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        # Compute mean
        if self.with_mean:
            self._mean = np.nanmean(data, axis=0)
        else:
            self._mean = np.zeros(data.shape[1])
 
        # Compute std
        if self.with_std:
            self._std = np.nanstd(data, axis=0)
            # Handle zero std (constant features)
            self._std[self._std == 0] = 1.0
        else:
            self._std = np.ones(data.shape[1])
 
        # Update state
        self._state = {
            'mean': self._mean.tolist(),
            'std': self._std.tolist(),
        }
 
        self._fitted = True
        return self
 
    def process(self, data: Any) -> np.ndarray:
        """Process data using fitted scaler.
 
        Args:
            data: Data to process (numpy array).
 
        Returns:
            Scaled data.
        """
        if not self._fitted:
            raise RuntimeError("StandardScaler must be fitted before processing.")
 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if self.copy:
            data = data.copy()
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        # Apply transformation
        if self.with_mean:
            data = data - self._mean
 
        if self.with_std:
            data = data / self._std
 
        return data
 
    def inverse_process(self, data: Any) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
 
        Args:
            data: Scaled data.
 
        Returns:
            Data in original scale.
        """
        if not self._fitted:
            raise RuntimeError("StandardScaler must be fitted before inverse processing.")
 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if self.copy:
            data = data.copy()
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        # Inverse transformation
        if self.with_std:
            data = data * self._std
 
        if self.with_mean:
            data = data + self._mean
 
        return data
 
    def get_mean(self) -> Optional[np.ndarray]:
        """Get computed mean values.
 
        Returns:
            Mean array or None if not fitted.
        """
        return self._mean
 
    def get_std(self) -> Optional[np.ndarray]:
        """Get computed standard deviation values.
 
        Returns:
            Std array or None if not fitted.
        """
        return self._std
 
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set processor state.
 
        Args:
            state: State dictionary containing 'mean' and 'std'.
        """
        self._mean = np.array(state['mean'])
        self._std = np.array(state['std'])
        self._state = state
        self._fitted = True
 
    def __repr__(self) -> str:
        return (
            f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std}, "
            f"fitted={self._fitted})"
        )