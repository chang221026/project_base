"""Variance threshold feature selector implementation."""
from typing import Any, Dict, List, Optional
 
import numpy as np
 
from lib.feature_selection import FEATURE_SELECTORS, BaseFeatureSelector
 
 
@FEATURE_SELECTORS.register('VarianceThreshold')
class VarianceThreshold(BaseFeatureSelector):
    """Feature selector that removes low-variance features.
 
    Removes all features whose variance doesn't meet the threshold.
 
    Config parameters:
        threshold: Variance threshold (default: 0.0).
            Features with variance <= threshold are removed.
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.0,
        **kwargs
    ):
        """Initialize VarianceThreshold.
 
        Args:
            config: Selector configuration dictionary.
            threshold: Variance threshold. Features with variance <= threshold are removed.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'threshold': threshold,
        })
        self.config.update(kwargs)
 
        self.threshold = self.config['threshold']
        self._variances = None
        self._selected_mask = None
        self._feature_names_in = []
 
    def fit(self, data: Any, target: Optional[Any] = None) -> 'VarianceThreshold':
        """Fit selector to data.
 
        Computes variance for each feature and determines which to keep.
 
        Args:
            data: Data to fit on (numpy array).
            target: Ignored, present for API consistency.
 
        Returns:
            Self for chaining.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        # Compute variance for each feature
        self._variances = np.nanvar(data, axis=0)
 
        # Create selection mask
        self._selected_mask = self._variances > self.threshold
 
        # Store feature names
        self._feature_names_in = [f'feature_{i}' for i in range(data.shape[1])]
 
        # Update selected features
        self._selected_features = [
            self._feature_names_in[i]
            for i in range(len(self._feature_names_in))
            if self._selected_mask[i]
        ]
 
        # Store feature scores (variance)
        self._feature_scores = {
            name: float(var)
            for name, var in zip(self._feature_names_in, self._variances)
        }
 
        self._fitted = True
        return self
 
    def select(self, data: Any, target: Optional[Any] = None) -> np.ndarray:
        """Select features from data.
 
        Args:
            data: Input data (numpy array).
            target: Ignored, present for API consistency.
 
        Returns:
            Data with selected features.
        """
        if not self._fitted:
            raise RuntimeError("VarianceThreshold must be fitted before selecting features.")
 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        return data[:, self._selected_mask]
 
    def fit_select(self, data: Any, target: Optional[Any] = None) -> np.ndarray:
        """Fit and select features in one step.
 
        Args:
            data: Data to fit and select from.
            target: Ignored, present for API consistency.
 
        Returns:
            Data with selected features.
        """
        return self.fit(data, target).select(data, target)
 
    def get_variances(self) -> Optional[np.ndarray]:
        """Get variance for each feature.
 
        Returns:
            Variance array or None if not fitted.
        """
        return self._variances
 
    def get_selected_indices(self) -> List[int]:
        """Get indices of selected features.
 
        Returns:
            List of selected feature indices.
        """
        if not self._fitted:
            return []
        return [i for i, selected in enumerate(self._selected_mask) if selected]
 
    def get_removed_features(self) -> List[str]:
        """Get names of removed features.
 
        Returns:
            List of removed feature names.
        """
        if not self._fitted:
            return []
        return [
            name for i, name in enumerate(self._feature_names_in)
            if not self._selected_mask[i]
        ]
 
    def __repr__(self) -> str:
        return f"VarianceThreshold(threshold={self.threshold}, fitted={self._fitted})"