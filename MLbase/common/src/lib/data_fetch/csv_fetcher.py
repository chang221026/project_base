"""CSV data fetcher implementation."""
import csv
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
 
import numpy as np
 
from lib.data_fetching import DATA_FETCHERS, BaseDataFetcher
 
 
@DATA_FETCHERS.register('CSVDataFetcher')
class CSVDataFetcher(BaseDataFetcher):
    """Data fetcher for CSV files.
 
    Loads data from CSV files with configurable options.
 
    Config parameters:
        delimiter: CSV delimiter character (default: ',').
        header: Row number for header (0-based, None for no header).
        target_column: Column name or index for target variable.
        feature_columns: List of column names or indices for features.
        skip_rows: Number of rows to skip at the beginning.
        encoding: File encoding (default: 'utf-8').
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        delimiter: str = ',',
        header: Optional[int] = 0,
        target_column: Optional[Union[str, int]] = None,
        feature_columns: Optional[List[Union[str, int]]] = None,
        skip_rows: int = 0,
        encoding: str = 'utf-8',
        **kwargs
    ):
        """Initialize CSVDataFetcher.
 
        Args:
            config: Fetcher configuration dictionary.
            delimiter: CSV delimiter character.
            header: Row number for header (0-based, None for no header).
            target_column: Column name or index for target variable.
            feature_columns: List of column names or indices for features.
            skip_rows: Number of rows to skip at the beginning.
            encoding: File encoding.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'delimiter': delimiter,
            'header': header,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'skip_rows': skip_rows,
            'encoding': encoding,
        })
        self.config.update(kwargs)
 
        self.delimiter = self.config['delimiter']
        self.header = self.config['header']
        self.target_column = self.config['target_column']
        self.feature_columns = self.config['feature_columns']
        self.skip_rows = self.config['skip_rows']
        self.encoding = self.config['encoding']
 
        self._header_names = None
        self._data = None
        self._features = None
        self._targets = None
 
    def fetch(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Fetch data from a CSV file.
 
        Args:
            source: Path to CSV file.
 
        Returns:
            Dictionary containing 'data', 'features', 'targets', and 'header'.
        """
        source = Path(source)
        if not self.validate_source(source):
            raise ValueError(f"Invalid source: {source}")
 
        data = []
        with open(source, 'r', encoding=self.encoding, newline='') as f:
            # Skip specified rows
            for _ in range(self.skip_rows):
                next(f)
 
            reader = csv.reader(f, delimiter=self.delimiter)
 
            # Read header if specified
            if self.header is not None:
                self._header_names = next(reader)
 
            for row in reader:
                data.append(row)
 
        self._data = np.array(data)
 
        # Separate features and targets if configured
        self._features, self._targets = self._extract_features_targets()
 
        # Update metadata
        self._metadata = {
            'source': str(source),
            'num_rows': len(data),
            'num_columns': len(data[0]) if data else 0,
            'header': self._header_names,
            'has_target': self._targets is not None,
        }
 
        return {
            'data': self._data,
            'features': self._features,
            'targets': self._targets,
            'header': self._header_names,
        }
 
    def _extract_features_targets(self) -> tuple:
        """Extract features and targets from data.
 
        Returns:
            Tuple of (features, targets) arrays.
        """
        if self._data is None or len(self._data) == 0:
            return None, None
 
        features = None
        targets = None
 
        # Get column indices
        if self._header_names is not None:
            col_indices = {name: i for i, name in enumerate(self._header_names)}
        else:
            col_indices = {i: i for i in range(self._data.shape[1])}
 
        # Extract features
        if self.feature_columns is not None:
            feature_indices = []
            for col in self.feature_columns:
                if isinstance(col, str):
                    feature_indices.append(col_indices.get(col, col_indices.get(col.lower(), None)))
                else:
                    feature_indices.append(col)
            feature_indices = [i for i in feature_indices if i is not None]
            features = self._data[:, feature_indices].astype(np.float32)
        else:
            # Use all columns except target
            all_indices = list(range(self._data.shape[1]))
            if self.target_column is not None:
                target_idx = col_indices.get(self.target_column, self.target_column)
                if isinstance(target_idx, int):
                    all_indices.remove(target_idx)
            features = self._data[:, all_indices].astype(np.float32)
 
        # Extract targets
        if self.target_column is not None:
            target_idx = col_indices.get(self.target_column, self.target_column)
            if isinstance(target_idx, int) or (isinstance(target_idx, str) and target_idx.isdigit()):
                target_idx = int(target_idx)
                targets = self._data[:, target_idx]
            else:
                targets = self._data[:, target_idx]
 
            # Convert targets to numeric type
            try:
                targets = targets.astype(np.int64)
            except (ValueError, TypeError):
                try:
                    targets = targets.astype(np.float32)
                except (ValueError, TypeError):
                    pass  # Keep as strings if conversion fails
 
        return features, targets
 
    def batch_fetch(
        self,
        sources: List[Union[str, Path]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        """Fetch data from multiple CSV files in batches.
 
        Args:
            sources: List of CSV file paths.
            batch_size: Number of files to fetch per batch.
 
        Yields:
            Batches of fetched data dictionaries.
        """
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i + batch_size]
            batch_data = []
            batch_features = []
            batch_targets = []
 
            for source in batch_sources:
                result = self.fetch(source)
                batch_data.append(result['data'])
                if result['features'] is not None:
                    batch_features.append(result['features'])
                if result['targets'] is not None:
                    batch_targets.append(result['targets'])
 
            yield {
                'data': batch_data,
                'features': np.vstack(batch_features) if batch_features else None,
                'targets': np.concatenate(batch_targets) if batch_targets else None,
            }
 
    def validate_source(self, source: Union[str, Path]) -> bool:
        """Validate CSV file source.
 
        Args:
            source: Data source to validate.
 
        Returns:
            True if valid CSV file, False otherwise.
        """
        source = Path(source)
        return source.exists() and source.suffix.lower() == '.csv'
 
    def get_features(self) -> Optional[np.ndarray]:
        """Get extracted features.
 
        Returns:
            Features array or None if not extracted.
        """
        return self._features
 
    def get_targets(self) -> Optional[np.ndarray]:
        """Get extracted targets.
 
        Returns:
            Targets array or None if not extracted.
        """
        return self._targets
 
    def __repr__(self) -> str:
        return (
            f"CSVDataFetcher(delimiter='{self.delimiter}', header={self.header}, "
            f"target_column={self.target_column})"
        )