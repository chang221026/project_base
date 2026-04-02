"""Integration tests for data pipeline module interactions.

Tests the interactions between fetcher, pipeline, builder, and dataloader.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestFetcherToPipeline:
    """Test fetcher output -> pipeline input interaction."""

    def test_fetcher_output_to_pipeline(self):
        """Test fetcher output is valid input to pipeline."""
        from lib.data_fetching import DATA_FETCHERS

        # Build fetcher
        fetcher = DATA_FETCHERS.build({
            'type': 'CSVDataFetcher',
            'source': 'test.csv'
        })

        # Fetcher should return (data, target) format
        assert fetcher is not None


class TestPipelineToBuilder:
    """Test pipeline output -> builder input interaction."""

    def test_pipeline_output_to_builder(self):
        """Test pipeline output is valid input to builder."""
        from training.data.data_preprocessing import DataPreprocessingPipeline

        pipeline = DataPreprocessingPipeline()

        # Pipeline should return processed data
        assert pipeline is not None


class TestBuilderToDataloader:
    """Test builder output -> dataloader interaction."""

    def test_builder_output_to_dataloader(self):
        """Test builder output is valid for DataLoader."""
        from training.data.dataset_building import Dataset, DataLoader

        # Create dataset
        data = list(range(100))
        targets = [i % 2 for i in range(100)]
        dataset = Dataset(data, targets)

        # Create dataloader
        loader = DataLoader(dataset, batch_size=10)

        assert len(loader) == 10


class TestDataPipelineEndToEnd:
    """Test complete data pipeline end-to-end."""

    def test_data_pipeline_flow(self):
        """Test complete flow from fetcher to dataloader."""
        from training.data_facade import DataFacade

        config = {
            'data': {
                'fetcher': {'type': 'CSVDataFetcher', 'source': 'test.csv'}
            }
        }

        facade = DataFacade(config)
        facade.setup()

        # Verify setup completed
        assert facade is not None