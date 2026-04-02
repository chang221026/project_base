"""Dataset building module.
 
Provides dataset construction and splitting functionality.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import random
 
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class Dataset:
    """Base dataset class."""
 
    def __init__(self, data: Any, targets: Optional[Any] = None,
                 transform: Optional[Callable] = None):
        """Initialize dataset.
 
        Args:
            data: Dataset data.
            targets: Optional targets/labels.
            transform: Optional transform to apply to samples.
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self._length = len(data) if hasattr(data, '__len__') else 0
 
    def __len__(self) -> int:
        """Get dataset length."""
        return self._length
 
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get sample at index."""
        if hasattr(self.data, '__getitem__'):
            sample = self.data[idx]
        else:
            sample = self.data
 
        if self.transform:
            sample = self.transform(sample)
 
        if self.targets is not None:
            if hasattr(self.targets, '__getitem__'):
                target = self.targets[idx]
            else:
                target = self.targets
            return sample, target
 
        return sample, None
 
    def split(self, ratios: List[float]) -> List['Dataset']:
        """Split dataset into subsets.
 
        Args:
            ratios: List of split ratios.
 
        Returns:
            List of dataset subsets.
 
        Raises:
            ValueError: If ratios do not sum to 1.0.
        """
        total = sum(ratios)
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
 
        n = len(self)
        indices = list(range(n))
        random.shuffle(indices)
 
        datasets = []
        start = 0
 
        for i, ratio in enumerate(ratios):
            # Last split takes all remaining indices to avoid data loss from int truncation
            if i == len(ratios) - 1:
                size = len(indices) - start
            else:
                size = int(n * ratio)
            subset_indices = indices[start:start + size]
            start += size
 
            subset_data = self._get_subset(self.data, subset_indices)
            subset_targets = None
            if self.targets is not None:
                subset_targets = self._get_subset(self.targets, subset_indices)
 
            datasets.append(Dataset(subset_data, subset_targets, self.transform))
 
        return datasets
 
    def _get_subset(self, data: Any, indices: List[int]) -> Any:
        """Get subset of data by indices."""
        if hasattr(data, 'iloc'):  # pandas DataFrame
            return data.iloc[indices]
        elif hasattr(data, '__getitem__'):
            return [data[i] for i in indices]
        return data
 
 
class DatasetBuilder:
    """Builder for creating and splitting datasets."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dataset builder.
 
        Args:
            config: Builder configuration containing:
                - train_ratio: Training set ratio
                - val_ratio: Validation set ratio
                - test_ratio: Test set ratio
                - shuffle: Whether to shuffle data
                - random_seed: Random seed for reproducibility
        """
        self.config = config or {}
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.test_ratio = self.config.get('test_ratio', 0.15)
        self.shuffle = self.config.get('shuffle', True)
        self.random_seed = self.config.get('random_seed', 42)
 
        random.seed(self.random_seed)
 
    def build(self, data: Any, targets: Optional[Any] = None,
              transform: Optional[Callable] = None) -> Dict[str, Dataset]:
        """Build train/val/test datasets.
 
        Args:
            data: Input data.
            targets: Optional targets.
            transform: Optional transform.
 
        Returns:
            Dictionary with 'train', 'val', 'test' datasets.
        """
        logger.info("Building datasets...")
 
        # Create full dataset
        full_dataset = Dataset(data, targets, transform)
 
        # Split dataset
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        splits = full_dataset.split(ratios)
 
        result = {
            'train': splits[0],
            'val': splits[1] if len(splits) > 1 else None,
            'test': splits[2] if len(splits) > 2 else None
        }
 
        logger.info(
            f"Dataset built: train={len(result['train'])}, "
            f"val={len(result['val']) if result['val'] else 0}, "
            f"test={len(result['test']) if result['test'] else 0}"
        )
 
        return result
 
    def build_train_val(self, data: Any, targets: Optional[Any] = None,
                        transform: Optional[Callable] = None) -> Tuple[Dataset, Dataset]:
        """Build train/val datasets only.
 
        Args:
            data: Input data.
            targets: Optional targets.
            transform: Optional transform.
 
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        datasets = self.build(data, targets, transform)
        return datasets['train'], datasets['val']
 
    def build_k_fold(self, data: Any, targets: Optional[Any] = None,
                     k: int = 5, transform: Optional[Callable] = None) -> List[Tuple[Dataset, Dataset]]:
        """Build k-fold cross-validation datasets.
 
        Args:
            data: Input data.
            targets: Optional targets.
            k: Number of folds.
            transform: Optional transform.
 
        Returns:
            List of (train_dataset, val_dataset) tuples.
        """
        logger.info(f"Building {k}-fold cross-validation datasets...")
 
        full_dataset = Dataset(data, targets, transform)
        n = len(full_dataset)
        indices = list(range(n))
 
        if self.shuffle:
            random.shuffle(indices)
 
        fold_size = n // k
        folds = []
 
        for i in range(k):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < k - 1 else n
 
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
 
            train_data = full_dataset._get_subset(data, train_indices)
            val_data = full_dataset._get_subset(data, val_indices)
 
            train_targets = None
            val_targets = None
            if targets is not None:
                train_targets = full_dataset._get_subset(targets, train_indices)
                val_targets = full_dataset._get_subset(targets, val_indices)
 
            train_dataset = Dataset(train_data, train_targets, transform)
            val_dataset = Dataset(val_data, val_targets, transform)
 
            folds.append((train_dataset, val_dataset))
 
        logger.info(f"Built {k} folds")
        return folds
 
 
class DataLoader:
    """Simple data loader for batching.
 
    Wraps Dataset to provide batch iteration for training.
    """
 
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False):
        """Initialize data loader.
 
        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
 
    def __iter__(self):
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
 
        if self.shuffle:
            random.shuffle(indices)
 
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
 
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
 
            # Gather samples and targets for batch
            samples = []
            targets = []
            for idx in batch_indices:
                sample, target = self.dataset[idx]
                samples.append(sample)
                targets.append(target)
 
            yield samples, targets
 
    def __len__(self) -> int:
        """Get number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
 
 
class DataBuilder:
    """Backward compatibility wrapper for building data loaders.
 
    This class provides a flexible way to create training and validation
    data loaders from configuration dictionaries.
    """
 
    @staticmethod
    def from_config(config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        """Build data loaders from configuration.
 
        Args:
            config: Configuration dictionary.
 
        Returns:
            Tuple of (train_loader, val_loader).
        """
        train_path = config.get('train_path')
        val_path = config.get('val_path')
        batch_size = config.get('batch_size', 32)
 
        train_loader = None
        val_loader = None
 
        if train_path:
            builder = DatasetBuilder(config)
            datasets = builder.build(train_path)
            train_loader = DataLoader(datasets['train'], batch_size=batch_size)
 
        if val_path and datasets.get('val'):
            val_loader = DataLoader(datasets['val'], batch_size=batch_size)
 
        return train_loader, val_loader
 
    @staticmethod
    def load_numpy(path: str, **kwargs) -> Dataset:
        """Load data from numpy file.
 
        Args:
            path: Path to numpy file.
            **kwargs: Additional arguments.
 
        Returns:
            Dataset instance.
        """
        import numpy as np
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                data = data.tolist()
        return Dataset(data, **kwargs)
 
    @staticmethod
    def load_csv(path: str, target: Optional[str] = None, **kwargs) -> Dataset:
        """Load data from CSV file.
 
        Args:
            path: Path to CSV file.
            target: Target column name.
            **kwargs: Additional arguments.
 
        Returns:
            Dataset instance.
        """
        import pandas as pd
        df = pd.read_csv(path)
        data = df.drop(columns=[target]) if target else df
        targets = df[target] if target else None
        return Dataset(data, targets, **kwargs)
 
    @staticmethod
    def load_json(path: str, **kwargs) -> Dataset:
        """Load data from JSON file.
 
        Args:
            path: Path to JSON file.
            **kwargs: Additional arguments.
 
        Returns:
            Dataset instance.
        """
        import json
        with open(path, 'r') as f:
            data = json.load(f)
 
        if isinstance(data, dict):
            if 'x' in data:
                data = data['x']
            elif 'data' in data:
                data = data['data']
        targets = kwargs.pop('target', None)
        if targets and isinstance(data, dict) and targets in data:
            targets = data[targets]
 
        return Dataset(data, targets, **kwargs)
 
    @classmethod
    def register_format(cls, format_name: str, loader_fn: Callable) -> None:
        """Register a custom data format loader.
 
        Args:
            format_name: Name of the format.
            loader_fn: Function that takes a path and returns a Dataset.
        """
        pass
 
 
def build_dataloaders(config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Build data loaders from configuration (backward compatibility).
 
    Args:
        config: Configuration dictionary.
 
    Returns:
        Tuple of (train_loader, val_loader).
    """
    return DataBuilder.from_config(config)