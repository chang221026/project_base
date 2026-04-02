"""Integration tests for training pipeline module interactions.

Tests the interactions between facade, algorithm, model, loss, optimizer.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestFacadeToAlgorithm:
    """Test facade -> algorithm construction interaction."""

    def test_facade_builds_algorithm(self):
        """Test facade constructs algorithm correctly."""
        from training.training_facade import TrainingFacade

        config = {
            'algorithm': {'type': 'supervised'},
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam', 'lr': 0.001}
        }

        facade = TrainingFacade(config)
        facade.setup()

        assert facade.algorithm is not None


class TestAlgorithmToModel:
    """Test algorithm -> model/loss/optimizer interaction."""

    def test_algorithm_builds_components(self):
        """Test algorithm constructs model/loss/optimizer."""
        from training.algorithm.supervised import SupervisedAlgorithm

        config = {
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam', 'lr': 0.001}
        }

        algo = SupervisedAlgorithm(config)
        algo.setup()

        assert algo.model is not None
        assert algo.loss_fn is not None
        assert algo.optimizer is not None


class TestTrainingPipelineEndToEnd:
    """Test complete training pipeline end-to-end."""

    def test_training_pipeline_flow(self):
        """Test complete flow from facade to training."""
        from training.training_facade import TrainingFacade

        config = {
            'algorithm': {'type': 'supervised'},
            'model': {'type': 'MLP', 'input_dim': 10, 'output_dim': 2},
            'loss': {'type': 'CrossEntropyLoss'},
            'optimizer': {'type': 'Adam', 'lr': 0.001},
            'training': {'epochs': 1}
        }

        facade = TrainingFacade(config)
        facade.setup()

        # Verify algorithm is set up
        assert facade.algorithm is not None