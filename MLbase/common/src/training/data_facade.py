"""Data facade module.
 
Provides a unified facade for data pipeline management, including
data fetching, preprocessing, dataset building.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
 
from utils.config_management import Config, instantiate
from utils.logger import get_logger
from training.data.data_preprocessing import DataPreprocessingPipeline
from training.data.dataset_building import Dataset, DatasetBuilder
 
 
logger = get_logger()
 
 
class DataFacade:
    """Facade for data pipeline management.
 
    This class provides a unified interface for all data-related operations:
    - Data fetching from various sources (via DataPreprocessingPipeline)
    - Data preprocessing pipeline (analysis, processing, feature engineering)
    - Dataset splitting (train/val/test)
 
    Architecture:
        DataFacade (入口)
            │
            ├──> DataPreprocessingPipeline (数据处理层)
            │       ├── fetcher: DATA_FETCHERS
            │       ├── analyzer: DATA_ANALYZERS
            │       ├── processors: DATA_PROCESSORS
            │       ├── constructors: FEATURE_CONSTRUCTORS
            │       ├── selectors: FEATURE_SELECTORS
            │       └── imbalance_handler: IMBALANCE_HANDLERS
            │
            └──> DatasetBuilder (数据集构建层)
                    ├── 装载数据
                    └── 划分 train/val/test
 
    Usage:
        # Standard usage with Trainer
        trainer = Trainer("config.yaml")
        # Trainer uses DataFacade internally to get train/val/test datasets
 
        # Direct usage
        facade = DataFacade(config)
        facade.setup()
        train_dataset, val_dataset, test_dataset = facade.get_datasets()
 
    Attributes:
        data_pipeline: Data preprocessing pipeline instance.
        dataset_builder: Dataset builder instance.
    """
 
    def __init__(self, config: Union[str, Path, Dict, Config], logger=None):
        """Initialize data facade from configuration.
 
        Args:
            config: Configuration file path, dictionary, or Config object.
            logger: Optional logger instance. If None, uses default logger.
        """
        self.logger = logger or get_logger()
        self.config = self._load_config(config)
 
        # Initialize data components
        self.data_pipeline: Optional[DataPreprocessingPipeline] = None
        self.dataset_builder: Optional[DatasetBuilder] = None
 
        # Cached datasets
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
 
        self.logger.info("DataFacade initialized")
 
    def _load_config(self, config: Union[str, Path, Dict, Config]) -> Config:
        """Load configuration.
 
        Args:
            config: Configuration in various formats.
 
        Returns:
            Config object.
        """
        from utils.config_management import load_config
 
        if isinstance(config, (str, Path)):
            return load_config(config)
        elif isinstance(config, dict):
            return Config.from_dict(config)
        elif isinstance(config, Config):
            return config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
 
    def setup(self) -> 'DataFacade':
        """Setup data pipeline from configuration.
 
        Returns:
            Self for chaining.
        """
        data_config = self.config.get('data', {})
        dataset_config = self.config.get('dataset', {})
 
        if data_config:
            # Setup preprocessing pipeline (数据处理层)
            self.data_pipeline = DataPreprocessingPipeline(data_config)
            self.data_pipeline.setup()
            self.logger.info("Data preprocessing pipeline setup complete")
 
            # Setup dataset builder (数据集构建层)
            self.dataset_builder = DatasetBuilder(dataset_config)
            self.logger.info("Dataset builder setup complete")
        else:
            self.logger.info("No data pipeline configuration found")
 
        return self
 
    def get_data_loaders(self,
                    train_loader: Optional['DataLoader'] = None,
                    val_loader: Optional['DataLoader'] = None,
                    test_loader: Optional['DataLoader'] = None) -> Tuple[Optional['DataLoader'], Optional['DataLoader'], Optional['DataLoader']]:
        """Get data loaders.
 
        If loaders are not provided, builds them from configuration.
        Returns DataLoaders ready for training.
 
        Args:
            train_loader: Training DataLoader. If None, builds from config.
            val_loader: Validation DataLoader. If None, builds from config.
            test_loader: Test DataLoader. If None, builds from config.
 
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        from .data.dataset_building import DataLoader
 
        # If all loaders provided, return them directly
        if train_loader is not None:
            return train_loader, val_loader, test_loader
 
        # Build from config if not provided
        if self.data_pipeline and self.dataset_builder:
            # Step 1: Fetch data using pipeline's fetcher
            data, targets = self._fetch_data()
 
            if data is None:
                self.logger.warning("No data fetched")
                return None, None, None
 
            # Step 2: Process data through pipeline
            self.logger.info("Processing data through pipeline...")
            data, targets = self.data_pipeline.run(data, targets, fit=True)
 
            # Step 3: Split data using dataset builder
            datasets = self.dataset_builder.build(data, targets)
 
            train_dataset = datasets.get('train')
            val_dataset = datasets.get('val')
            test_dataset = datasets.get('test')
 
            # Step 4: Wrap datasets in DataLoaders
            batch_size = self.config.get('training', {}).get('batch_size', 32)
 
            self._train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
            self._val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
            self._test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
 
            self.logger.info(
                f"Data loaders built: train={len(train_dataset) if train_dataset else 0}, "
                f"val={len(val_dataset) if val_dataset else 0}, "
                f"test={len(test_dataset) if test_dataset else 0}"
            )
 
            return self._train_loader, self._val_loader, self._test_loader
 
        return None, None, None
 
        return None, None, None
 
    def _fetch_data(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Fetch data from configured source.
 
        Uses the pipeline's fetcher to fetch data.
 
        Returns:
            Tuple of (data, targets) or (None, None) if not configured.
        """
        if not self.data_pipeline or not self.data_pipeline.fetcher:
            return None, None
 
        fetcher_config = self.config.get('data', {}).get('fetcher', {})
        source = fetcher_config.get('source')
 
        if not source:
            return None, None
 
        self.logger.info(f"Fetching data from: {source}")
        fetched = self.data_pipeline.fetcher.fetch(source)
 
        # Use safe key lookup (avoid 'or' with numpy arrays)
        data = fetched.get('features')
        if data is None:
            data = fetched.get('data')
        if data is None:
            data = fetched.get('X')
 
        targets = fetched.get('targets')
        if targets is None:
            targets = fetched.get('labels')
        if targets is None:
            targets = fetched.get('y')
 
        return data, targets
 
    def get_pipeline(self) -> Optional[DataPreprocessingPipeline]:
        """Get data preprocessing pipeline.
 
        Returns:
            Data preprocessing pipeline or None if not configured.
        """
        return self.data_pipeline
 
    def get_dataset_builder(self) -> Optional[DatasetBuilder]:
        """Get dataset builder.
 
        Returns:
            Dataset builder or None if not configured.
        """
        return self.dataset_builder
 
 
def create_data_facade(config: Union[str, Dict, Config]) -> DataFacade:
    """Createand setup data facade.
 
    Args:
        config: Configuration file path or dictionary.
 
    Returns:
        Configured DataFacade instance.
 
    Example:
        >>> facade = create_data_facade("config.yaml")
        >>> train_ds, val_ds, test_ds = facade.get_datasets()
    """
    facade = DataFacade(config)
    facade.setup()
    return facade