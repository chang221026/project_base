"""Unit tests for data pipeline components.

Migrated from tests/ut/test_training_data_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDataPreprocessingPipeline:
    """Test DataPreprocessingPipeline functionality."""

    def test_pipeline_exists(self):
        """Test pipeline exists."""
        from training.data.data_preprocessing import DataPreprocessingPipeline
        assert DataPreprocessingPipeline is not None


class TestDatasetBuilder:
    """Test DatasetBuilder functionality."""

    def test_dataset_builder_exists(self):
        """Test builder exists."""
        from training.data.dataset_building import DatasetBuilder
        assert DatasetBuilder is not None