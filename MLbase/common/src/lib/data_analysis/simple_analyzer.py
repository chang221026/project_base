"""Simple data analyzer implementation."""
from typing import Any, Dict, List, Optional
 
import numpy as np
 
from lib.data_analysis import DATA_ANALYZERS
from lib.data_analysis.base import BaseDataAnalyzer, DataProfile
 
 
@DATA_ANALYZERS.register('SimpleDataAnalyzer')
class SimpleDataAnalyzer(BaseDataAnalyzer):
    """Simple data analyzer for basic statistics.
 
    Computes basic statistics including mean, standard deviation,
    missing values, and distributions.
 
    Config parameters:
        compute_correlations: Whether to compute correlations (default: False).
        percentiles: Percentiles to compute (default: [25, 50, 75]).
        max_unique_values: Maximum unique values for distribution analysis.
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        compute_correlations: bool = False,
        percentiles: Optional[List[float]] = None,
        max_unique_values: int = 20,
        **kwargs
    ):
        """Initialize SimpleDataAnalyzer.
 
        Args:
            config: Analyzer configuration dictionary.
            compute_correlations: Whether to compute correlations.
            percentiles: Percentiles to compute.
            max_unique_values: Maximum unique values for distribution analysis.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'compute_correlations': compute_correlations,
            'percentiles': percentiles or [25, 50, 75],
            'max_unique_values': max_unique_values,
        })
        self.config.update(kwargs)
 
        self.compute_correlations = self.config['compute_correlations']
        self.percentiles = self.config['percentiles']
        self.max_unique_values = self.config['max_unique_values']
 
    def analyze(self, data: Any) -> DataProfile:
        """Analyze data and generate profile.
 
        Args:
            data: Data to analyze (numpy array or list).
 
        Returns:
            DataProfile with analysis results.
        """
        # Convert to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        self.profile = DataProfile()
 
        # Basic info
        self.profile.shape = data.shape
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        # Column info
        num_cols = data.shape[1] if data.ndim > 1 else 1
        self.profile.columns = [f'col_{i}' for i in range(num_cols)]
        self.profile.dtypes = {f'col_{i}': str(data[:, i].dtype) for i in range(num_cols)}
 
        # Missing values
        self.profile.missing_values = self._compute_missing_values(data)
 
        # Statistics
        self.profile.statistics = self.compute_statistics(data)
 
        # Distributions
        self.profile.distributions = self._compute_distributions(data)
 
        # Correlations (optional)
        if self.compute_correlations:
            self.profile.correlations = self._compute_correlations(data)
 
        return self.profile
 
    def compute_statistics(self, data: Any, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute statistics for data.
 
        Args:
            data: Data to analyze.
            columns: Optional list of columns to analyze.
 
        Returns:
            Statistics dictionary.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        statistics = {}
 
        for i in range(data.shape[1]):
            col_name = f'col_{i}'
            if columns and col_name not in columns:
                continue
 
            col_data = data[:, i]
 
            # Check if numeric
            if np.issubdtype(col_data.dtype, np.number):
                col_stats = {
                    'mean': float(np.nanmean(col_data)),
                    'std': float(np.nanstd(col_data)),
                    'min': float(np.nanmin(col_data)),
                    'max': float(np.nanmax(col_data)),
                    'median': float(np.nanmedian(col_data)),
                }
 
                # Percentiles
                for p in self.percentiles:
                    col_stats[f'percentile_{int(p)}'] = float(np.nanpercentile(col_data, p))
 
                # Missing count
                col_stats['missing_count'] = int(np.sum(np.isnan(col_data)))
                col_stats['missing_ratio'] = float(np.mean(np.isnan(col_data)))
 
            else:
                # Non-numeric column
                unique_vals, counts = np.unique(col_data, return_counts=True)
                col_stats = {
                    'unique_count': len(unique_vals),
                    'most_common': str(unique_vals[np.argmax(counts)]),
                    'most_common_count': int(np.max(counts)),
                }
 
            statistics[col_name] = col_stats
 
        return statistics
 
    def _compute_missing_values(self, data: np.ndarray) -> Dict[str, int]:
        """Compute missing values for each column.
 
        Args:
            data: Data array.
 
        Returns:
            Dictionary of column name to missing count.
        """
        missing = {}
        for i in range(data.shape[1]):
            col_data = data[:, i]
            if np.issubdtype(col_data.dtype, np.number):
                missing[f'col_{i}'] = int(np.sum(np.isnan(col_data)))
            else:
                # For non-numeric, check for empty strings or 'NA'
                missing[f'col_{i}'] = int(np.sum((col_data == '') | (col_data == 'NA') | (col_data == 'null')))
        return missing
 
    def _compute_distributions(self, data: np.ndarray) -> Dict[str, Dict]:
        """Compute value distributions for each column.
 
        Args:
            data: Data array.
 
        Returns:
            Dictionary of column name to distribution info.
        """
        distributions = {}
        for i in range(data.shape[1]):
            col_data = data[:, i]
            unique_vals, counts = np.unique(col_data, return_counts=True)
 
            if len(unique_vals) <= self.max_unique_values:
                distributions[f'col_{i}'] = {
                    'type': 'categorical',
                    'values': {str(v): int(c) for v, c in zip(unique_vals, counts)},
                }
            else:
                # For high-cardinality numeric columns, compute histogram
                if np.issubdtype(col_data.dtype, np.number):
                    hist, bin_edges = np.histogram(col_data[~np.isnan(col_data)], bins=10)
                    distributions[f'col_{i}'] = {
                        'type': 'numeric',
                        'histogram': {
                            'counts': hist.tolist(),
                            'bin_edges': bin_edges.tolist(),
                        },
                    }
                else:
                    distributions[f'col_{i}'] = {
                        'type': 'high_cardinality',
                        'unique_count': len(unique_vals),
                    }
 
        return distributions
 
    def _compute_correlations(self, data: np.ndarray) -> Dict[str, Dict]:
        """Compute correlations between numeric columns.
 
        Args:
            data: Data array.
 
        Returns:
            Dictionary of correlation matrix.
        """
        correlations = {}
 
        # Only for numeric columns
        numeric_cols = []
        for i in range(data.shape[1]):
            if np.issubdtype(data[:, i].dtype, np.number):
                numeric_cols.append(i)
 
        if len(numeric_cols) < 2:
            return correlations
 
        # Compute correlation matrix
        numeric_data = data[:, numeric_cols]
        try:
            corr_matrix = np.corrcoef(numeric_data.T)
            for i, col_i in enumerate(numeric_cols):
                correlations[f'col_{col_i}'] = {
                    f'col_{col_j}': float(corr_matrix[i, j])
                    for j, col_j in enumerate(numeric_cols)
                }
        except Exception:
            pass
 
        return correlations
 
    def detect_anomalies(self, data: Any) -> Dict[str, Any]:
        """Detect anomalies in data using IQR method.
 
        Args:
            data: Data to analyze.
 
        Returns:
            Anomaly detection results.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
 
        if data.ndim == 1:
            data = data.reshape(-1, 1)
 
        anomalies = {}
 
        for i in range(data.shape[1]):
            col_data = data[:, i]
 
            if np.issubdtype(col_data.dtype, np.number):
                q1 = np.nanpercentile(col_data, 25)
                q3 = np.nanpercentile(col_data, 75)
                iqr = q3 - q1
 
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
 
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_count = np.sum(outlier_mask)
 
                anomalies[f'col_{i}'] = {
                    'outlier_count': int(outlier_count),
                    'outlier_ratio': float(outlier_count / len(col_data)),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                }
 
        return anomalies
 
    def __repr__(self) -> str:
        return f"SimpleDataAnalyzer(compute_correlations={self.compute_correlations})"