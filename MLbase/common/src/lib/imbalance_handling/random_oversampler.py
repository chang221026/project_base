"""Random oversampler for handling class imbalance."""
import random
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
 
from lib.imbalance_handling import IMBALANCE_HANDLERS, BaseImbalanceHandler
 
 
@IMBALANCE_HANDLERS.register('RandomOverSampler')
class RandomOverSampler(BaseImbalanceHandler):
    """Random oversampler for handling class imbalance.
 
    Randomly duplicates samples from minority classes to balance the dataset.
 
    Config parameters:
        sampling_strategy: Sampling strategy (default: 'auto').
            - 'auto': Balance all classes to the majority class size.
            - dict: {class_label: target_count} for specific class targets.
            - float: Target ratio of minority to majority class.
        random_state: Random seed for reproducibility (default: None).
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None,
        **kwargs
    ):
        """Initialize RandomOverSampler.
 
        Args:
            config: Handler configuration dictionary.
            sampling_strategy: Sampling strategy ('auto', dict, or float).
            random_state: Random seed for reproducibility.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'sampling_strategy': sampling_strategy,
            'random_state': random_state,
        })
        self.config.update(kwargs)
 
        self.sampling_strategy = self.config['sampling_strategy']
        self.random_state = self.config['random_state']
 
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
 
        self._class_counts = {}
        self._sample_indices = {}
 
    def fit(self, data: Any, target: Any) -> 'RandomOverSampler':
        """Fit handler to data.
 
        Computes class distribution and sampling strategy.
 
        Args:
            data: Data to fit on.
            target: Target labels.
 
        Returns:
            Self for chaining.
        """
        if not isinstance(target, np.ndarray):
            target = np.array(target)
 
        # Compute class distribution
        unique_classes, counts = np.unique(target, return_counts=True)
        self._class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))
        self._class_counts = self._class_distribution.copy()
 
        # Store sample indices for each class
        self._sample_indices = {}
        for cls in unique_classes:
            self._sample_indices[cls] = np.where(target == cls)[0]
 
        self._fitted = True
        return self
 
    def handle(self, data: Any, target: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance by oversampling.
 
        Args:
            data: Input data.
            target: Target labels.
 
        Returns:
            Tuple of (balanced_data, balanced_target).
        """
        if not self._fitted:
            self.fit(data, target)
 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if not isinstance(target, np.ndarray):
            target = np.array(target)
 
        # Determine target counts based on sampling strategy
        target_counts = self._compute_target_counts()
 
        # Generate resampled indices
        resampled_indices = []
 
        for cls, target_count in target_counts.items():
            original_indices = self._sample_indices[cls]
            original_count = len(original_indices)
 
            if target_count <= original_count:
                # No oversampling needed, use original indices
                resampled_indices.extend(original_indices[:target_count].tolist())
            else:
                # Use all original indices
                resampled_indices.extend(original_indices.tolist())
 
                # Oversample by random selection with replacement
                n_samples_needed = target_count - original_count
                oversample_indices = np.random.choice(
                    original_indices,
                    size=n_samples_needed,
                    replace=True
                )
                resampled_indices.extend(oversample_indices.tolist())
 
        # Shuffle indices
        random.shuffle(resampled_indices)
 
        # Create balanced dataset
        balanced_data = data[resampled_indices]
        balanced_target = target[resampled_indices]
 
        return balanced_data, balanced_target
 
    def fit_handle(self, data: Any, target: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and handle imbalance in one step.
 
        Args:
            data: Data to fit and handle.
            target: Target labels.
 
        Returns:
            Tuple of (balanced_data, balanced_target).
        """
        return self.fit(data, target).handle(data, target)
 
    def _compute_target_counts(self) -> Dict[Any, int]:
        """Compute target sample counts for each class.
 
        Returns:
            Dictionary mapping class labels to target counts.
        """
        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
 
        max_count = max(self._class_counts.values())
 
        if self.sampling_strategy == 'auto':
            # Balance all classes to majority count
            return {cls: max_count for cls in self._class_counts.keys()}
 
        elif isinstance(self.sampling_strategy, float):
            # Target ratio of minority to majority
            return {
                cls: int(max_count * self.sampling_strategy)
                for cls in self._class_counts.keys()
            }
 
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
 
    def get_oversampling_info(self) -> Dict[str, Any]:
        """Get information about the oversampling.
 
        Returns:
            Dictionary with original and target class counts.
        """
        if not self._fitted:
            return {}
 
        target_counts = self._compute_target_counts()
 
        return {
            'original_counts': self._class_counts.copy(),
            'target_counts': target_counts,
            'total_original': sum(self._class_counts.values()),
            'total_resampled': sum(target_counts.values()),
        }
 
    def __repr__(self) -> str:
        return (
            f"RandomOverSampler(sampling_strategy={self.sampling_strategy}, "
            f"random_state={self.random_state}, fitted={self._fitted})"
        )