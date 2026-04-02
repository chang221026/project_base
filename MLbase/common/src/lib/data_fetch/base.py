"""Base data fetcher class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, List, Union
from pathlib import Path
 
 
class BaseDataFetcher(ABC):
    """Base class for all data fetchers.
 
    Data fetchers are responsible for loading raw data from various sources.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize data fetcher.
 
        Args:
            config: Fetcher configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.config = config or {}
        self.config.update(kwargs)
        self._data = None
        self._metadata = {}
 
    @abstractmethod
    def fetch(self, source: Union[str, Path]) -> Any:
        """Fetch data from source.
 
        Args:
            source: Data source (file path, URL, etc.).
 
        Returns:
            Fetched data.
        """
        pass
 
    @abstractmethod
    def batch_fetch(self, sources: List[Union[str, Path]], batch_size: int = 32) -> Iterator:
        """Fetch data in batches.
 
        Args:
            sources: List of data sources.
            batch_size: Number of sources to fetch per batch.
 
        Yields:
            Batches of fetched data.
        """
        pass
 
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about fetched data.
 
        Returns:
            Metadata dictionary.
        """
        return self._metadata.copy()
 
    def validate_source(self, source: Union[str, Path]) -> bool:
        """Validate data source.
 
        Args:
            source: Data source to validate.
 
        Returns:
            True if valid, False otherwise.
        """
        return True
 
    def get_config(self) -> Dict[str, Any]:
        """Get fetcher configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()