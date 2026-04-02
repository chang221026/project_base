"""Unit tests for DATA_FETCHERS registry.

Migrated from tests/ut/test_lib_data_fetching_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDataFetchersRegistry:
    """Test DATA_FETCHERS registry functionality."""

    def test_data_fetchers_registry_exists(self):
        """Test DATA_FETCHERS registry exists."""
        from lib.data_fetching import DATA_FETCHERS
        assert DATA_FETCHERS is not None

    def test_csv_fetcher_registration(self):
        """Test CSVDataFetcher is registered."""
        from lib.data_fetching import DATA_FETCHERS

        fetcher = DATA_FETCHERS.get('CSVDataFetcher')
        assert fetcher is not None

    def test_build_csv_fetcher(self):
        """Test building CSVDataFetcher from config."""
        from lib.data_fetching import DATA_FETCHERS

        fetcher = DATA_FETCHERS.build({
            'type': 'CSVDataFetcher',
            'source': 'test.csv'
        })
        assert fetcher is not None