"""Polynomial features constructor implementation."""
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional
 
import numpy as np
 
from lib.feature_construction import FEATURE_CONSTRUCTORS, BaseFeatureConstructor
 
 
@FEATURE_CONSTRUCTORS.register('PolynomialFeatures')
class PolynomialFeatures(BaseFeatureConstructor):
    """Polynomial feature constructor.
 
    Generates polynomial and interaction features.
 
    Config parameters:
        degree: Maximum degree of polynomial features (default: 2).
        include_bias: Whether to include bias (constant) term (default: False).
        interaction_only: Whether to only include interaction features (default: False).
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        **kwargs
    ):
        """Initialize PolynomialFeatures.
 
        Args:
            config: Constructor configuration dictionary.
            degree: Maximum degree of polynomial features.
            include_bias: Whether to include bias term.
            interaction_only: Whether to only include interaction features.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'degree': degree,
            'include_bias': include_bias,
            'interaction_only': interaction_only,
        })
        self.config.update(kwargs)
 
        self.degree = self.config['degree']
        self.include_bias = self.config['include_bias']
        self.interaction_only = self.config['interaction_only']
 
        self._n_input_features = 0
        self._n_output_features = 0
        self._powers = []
 
    def fit(self, data: Any) -> 'PolynomialFeatures':
        """Fit constructor to data.
 
        Computes the number of output features and power combinations.
 
        Args:
            data: Data to fit on (numpy array).
 
        Returns:
            Self for chaining.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        self._n_input_features = data.shape[1]
        self._powers = self._compute_powers()
        self._n_output_features = len(self._powers)
        self._feature_names = self._generate_feature_names()
 
        self._fitted = True
        return self
 
    def _compute_powers(self) -> List[tuple]:
        """Compute power combinations for each output feature.
 
        Returns:
            List of power tuples.
        """
        powers = []
 
        # Bias term (all zeros)
        if self.include_bias:
            powers.append(tuple([0] * self._n_input_features))
 
        # Start from degree 1 if interaction_only, else from degree 0
        start_degree = 1 if self.interaction_only else 0
 
        for degree in range(start_degree, self.degree + 1):
            # Generate all combinations of powers that sum to degree
            for comb in combinations_with_replacement(range(self._n_input_features), degree):
                if self.interaction_only and degree > 1:
                    # Only include if all selected features are distinct
                    if len(set(comb)) != len(comb):
                        continue
 
                # Convert combination to power tuple
                power = [0] * self._n_input_features
                for idx in comb:
                    power[idx] += 1
 
                # Skip degree 0 (bias) if already added
                if degree == 0 and self.include_bias:
                    continue
 
                powers.append(tuple(power))
 
        return powers
 
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names based on powers.
 
        Returns:
            List of feature names.
        """
        names = []
 
        for power in self._powers:
            if sum(power) == 0:
                names.append('1')  # Bias term
            else:
                parts = []
                for i, p in enumerate(power):
                    if p > 0:
                        if p == 1:
                            parts.append(f'x{i}')
                        else:
                            parts.append(f'x{i}^{p}')
                names.append(' '.join(parts))
 
        return names
 
    def construct(self, data: Any) -> np.ndarray:
        """Construct polynomial features from data.
 
        Args:
            data: Input data (numpy array).
 
        Returns:
            Data with polynomial features.
        """
        if not self._fitted:
            raise RuntimeError("PolynomialFeatures must be fitted before constructing features.")
 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        n_samples = data.shape[0]
 
        # Generate polynomial features
        result = np.zeros((n_samples, self._n_output_features))
 
        for i, power in enumerate(self._powers):
            feature = np.ones(n_samples)
            for j, p in enumerate(power):
                if p > 0:
                    feature = feature * (data[:, j] ** p)
            result[:, i] = feature
 
        return result
 
    def fit_construct(self, data: Any) -> np.ndarray:
        """Fit and construct features in one step.
 
        Args:
            data: Data to fit and construct features from.
 
        Returns:
            Data with polynomial features.
        """
        return self.fit(data).construct(data)
 
    def get_feature_names(self) -> List[str]:
        """Get names of constructed features.
 
        Returns:
            List of feature names.
        """
        return self._feature_names.copy()
 
    def get_n_output_features(self) -> int:
        """Get number of output features.
 
        Returns:
            Number of output features.
        """
        return self._n_output_features
 
    def __repr__(self) -> str:
        return (
            f"PolynomialFeatures(degree={self.degree}, include_bias={self.include_bias}, "
            f"interaction_only={self.interaction_only}, fitted={self._fitted})"
        )