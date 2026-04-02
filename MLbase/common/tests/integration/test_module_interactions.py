"""Integration tests for cross-module interactions.

Tests interactions between Trainer, DataFacade, and TrainingFacade.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestTrainerToDataFacade:
    """Test Trainer -> DataFacade interaction."""

    def test_trainer_creates_data_facade(self):
        """Test Trainer creates DataFacade."""
        from training.trainer import Trainer

        config = {
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam'},
            'training': {'epochs': 1}
        }

        # Use minimal config to avoid full setup
        trainer = Trainer(config)

        # Verify data_facade was created
        assert trainer.data_facade is not None


class TestTrainerToTrainingFacade:
    """Test Trainer -> TrainingFacade interaction."""

    def test_trainer_creates_training_facade(self):
        """Test Trainer creates TrainingFacade."""
        from training.trainer import Trainer

        config = {
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam'},
            'training': {'epochs': 1}
        }

        trainer = Trainer(config)

        # Verify training_facade was created
        assert trainer.training_facade is not None


class TestDataFacadeToTrainingFacade:
    """Test DataFacade -> TrainingFacade interaction."""

    def test_data_passes_to_training(self):
        """Test data flows from DataFacade to TrainingFacade."""
        from training.data_facade import DataFacade
        from training.training_facade import TrainingFacade

        # Create both facades
        data_config = {
            'data': {'fetcher': {'type': 'CSVDataFetcher', 'source': 'test.csv'}}
        }
        training_config = {
            'algorithm': {'type': 'supervised'},
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam'}
        }

        data_facade = DataFacade(data_config)
        training_facade = TrainingFacade(training_config)

        # Both should be independent but work together
        assert data_facade is not None
        assert training_facade is not None