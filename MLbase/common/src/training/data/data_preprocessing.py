"""Data preprocessing pipeline.
 
Provides a unified pipeline for data preprocessing.
"""
from typing import Any, Dict, List, Optional, Callable
 
from utils.logger import get_logger
from lib.data_fetching import DATA_FETCHERS
from lib.data_analysis import DATA_ANALYZERS
from lib.data_processing import DATA_PROCESSORS
from lib.feature_construction import FEATURE_CONSTRUCTORS
from lib.feature_selection import FEATURE_SELECTORS
from lib.imbalance_handling import IMBALANCE_HANDLERS
 
 
logger = get_logger()
 
 
class DataPreprocessingPipeline:
    """Data preprocessing pipeline.
 
    Orchestrates data fetching, analysis, processing, feature engineering,
    and imbalance handling.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing pipeline.
 
        Args:
            config: Pipeline configuration containing:
                - fetcher: Data fetcher configuration
                - analyzer: Data analyzer configuration
                - processors: List of data processor configurations
                - constructors: List of feature constructor configurations
                - selectors: List of feature selector configurations
                - imbalance_handler: Imbalance handler configuration
        """
        self.config = config or {}
        self.fetcher = None
        self.analyzer = None
        self.processors: List = []
        self.constructors: List = []
        self.selectors: List = []
        self.imbalance_handler = None
        self._fitted = False
 
    def setup(self) -> 'DataPreprocessingPipeline':
        """Setup pipeline components from config."""
        # Setup fetcher
        fetcher_config = self.config.get('fetcher')
        if fetcher_config:
            self.fetcher = DATA_FETCHERS.build(fetcher_config)
            logger.info(f"Data fetcher: {type(self.fetcher).__name__}")
 
        # Setup analyzer
        analyzer_config = self.config.get('analyzer')
        if analyzer_config:
            self.analyzer = DATA_ANALYZERS.build(analyzer_config)
            logger.info(f"Data analyzer: {type(self.analyzer).__name__}")
 
        # Setup processors
        processor_configs = self.config.get('processors', [])
        for proc_config in processor_configs:
            processor = DATA_PROCESSORS.build(proc_config)
            self.processors.append(processor)
            logger.info(f"Data processor: {type(processor).__name__}")
 
        # Setup feature constructors
        constructor_configs = self.config.get('constructors', [])
        for cons_config in constructor_configs:
            constructor = FEATURE_CONSTRUCTORS.build(cons_config)
            self.constructors.append(constructor)
            logger.info(f"Feature constructor: {type(constructor).__name__}")
 
        # Setup feature selectors
        selector_configs = self.config.get('selectors', [])
        for sel_config in selector_configs:
            selector = FEATURE_SELECTORS.build(sel_config)
            self.selectors.append(selector)
            logger.info(f"Feature selector: {type(selector).__name__}")
 
        # Setup imbalance handler
        imbalance_config = self.config.get('imbalance_handler')
        if imbalance_config:
            self.imbalance_handler = IMBALANCE_HANDLERS.build(imbalance_config)
            logger.info(f"Imbalance handler: {type(self.imbalance_handler).__name__}")
 
        return self
 
    def run(self, data: Any, target: Optional[Any] = None, fit: bool = True) -> tuple:
        """Run preprocessing pipeline.
 
        Args:
            data: Input data.
            target: Optional target labels.
            fit: Whether to fit transformers on data.
 
        Returns:
            Tuple of (processed_data, target).
        """
        logger.info("Starting data preprocessing pipeline")
 
        # Analyze data
        if self.analyzer:
            logger.info("Analyzing data...")
            profile = self.analyzer.analyze(data)
            logger.info(f"Data profile: {profile.to_dict()}")
 
        # Apply data processors
        for processor in self.processors:
            logger.info(f"Applying {type(processor).__name__}...")
            if fit:
                data = processor.fit_process(data)
            else:
                data = processor.process(data)
 
        # Apply feature constructors
        for constructor in self.constructors:
            logger.info(f"Constructing features with {type(constructor).__name__}...")
            if fit:
                data = constructor.fit_construct(data)
            else:
                data = constructor.construct(data)
 
        # Apply feature selectors
        for selector in self.selectors:
            logger.info(f"Selecting features with {type(selector).__name__}...")
            if fit:
                data = selector.fit_select(data, target)
            else:
                data = selector.select(data, target)
 
        # Handle imbalance
        if self.imbalance_handler and target is not None:
            logger.info(f"Handling imbalance with {type(self.imbalance_handler).__name__}...")
            if fit:
                data, target = self.imbalance_handler.fit_handle(data, target)
            else:
                data, target = self.imbalance_handler.handle(data, target)
 
        if fit:
            self._fitted = True
 
        logger.info("Data preprocessing complete")
        return data, target
 
    def fit(self, data: Any, target: Optional[Any] = None) -> 'DataPreprocessingPipeline':
        """Fit pipeline to data.
 
        Args:
            data: Training data.
            target: Optional target labels.
 
        Returns:
            Self for chaining.
        """
        self.run(data, target, fit=True)
        return self
 
    def transform(self, data: Any, target: Optional[Any] = None) -> tuple:
        """Transform data using fitted pipeline.
 
        Args:
            data: Data to transform.
            target: Optional target labels.
 
        Returns:
            Tuple of (transformed_data, target).
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        return self.run(data, target, fit=False)
 
    def fit_transform(self, data: Any, target: Optional[Any] = None) -> tuple:
        """Fit and transform data.
 
        Args:
            data: Data to fit and transform.
            target: Optional target labels.
 
        Returns:
            Tuple of (transformed_data, target).
        """
        return self.fit(data, target).transform(data, target)
 
    def save(self, filepath: str) -> None:
        """Save pipeline state."""
        raise NotImplementedError("Save not implemented")
 
    def load(self, filepath: str) -> None:
        """Load pipeline state."""
        raise NotImplementedError("Load not implemented")