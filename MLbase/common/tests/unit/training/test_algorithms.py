"""Unit tests for algorithm interface.

Migrated from tests/ut/test_training_algorithm_*.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestAlgorithmInterface:
    """Test BaseAlgorithm interface functionality."""

    def test_base_algorithm_exists(self):
        """Test BaseAlgorithm exists."""
        from training.algorithm.base import BaseAlgorithm
        assert BaseAlgorithm is not None

    def test_algorithm_setup(self):
        """Test algorithm setup with concrete implementation."""
        from training.algorithm.base import BaseAlgorithm

        class TestAlgorithm(BaseAlgorithm):
            def setup(self):
                self.model = MagicMock()

            def train_step(self, batch):
                return {'loss': 0.5}

            def val_step(self, batch):
                return {'val_loss': 0.5}

        algo = TestAlgorithm()
        algo.setup()
        assert algo.model is not None

    def test_algorithm_fit(self):
        """Test algorithm fit with concrete implementation."""
        from training.algorithm.base import BaseAlgorithm

        class TestAlgorithm(BaseAlgorithm):
            def setup(self):
                self.model = MagicMock()
                self.optimizer = MagicMock()

            def train_step(self, batch):
                return {'loss': 0.5}

            def val_step(self, batch):
                return {'val_loss': 0.5}

        algo = TestAlgorithm()
        algo.setup()

        train_loader = MagicMock()
        train_loader.__len__ = lambda self: 2
        train_loader.__iter__ = lambda self: iter([(MagicMock(), MagicMock()) for _ in range(2)])

        history = algo.fit(train_loader, epochs=1)
        assert 'train' in history


class TestSupervisedAlgorithm:
    """Test SupervisedAlgorithm functionality."""

    def test_supervised_algorithm_exists(self):
        """Test SupervisedAlgorithm exists."""
        from training.algorithm.supervised import SupervisedAlgorithm
        assert SupervisedAlgorithm is not None


class TestRLAlgorithm:
    """Test RLAlgorithm functionality."""

    def test_rl_algorithm_exists(self):
        """Test RLAlgorithm exists."""
        from training.algorithm.rl import RLAlgorithm
        assert RLAlgorithm is not None

    def test_ppo_exists(self):
        """Test PPO exists."""
        from training.algorithm.rl import PPO
        assert PPO is not None

    def test_sac_exists(self):
        """Test SAC exists."""
        from training.algorithm.rl import SAC
        assert SAC is not None