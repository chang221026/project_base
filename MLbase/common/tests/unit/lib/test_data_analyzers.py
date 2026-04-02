"""Unit tests for DATA_ANALYZERS registry."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDataAnalyzersRegistry:
    """Test DATA_ANALYZERS registry functionality."""

    def test_data_analyzers_registry_exists(self):
        """Test DATA_ANALYZERS registry exists."""
        from lib.data_analysis import DATA_ANALYZERS
        assert DATA_ANALYZERS is not None