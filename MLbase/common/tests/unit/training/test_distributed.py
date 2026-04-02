"""Unit tests for distributed training.

Migrated from tests/ut/test_training_distributed_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDistributedEngine:
    """Test DistributedEngine functionality."""

    def test_distributed_engine_exists(self):
        """Test DistributedEngine exists."""
        from training.distributed import DistributedEngine
        assert DistributedEngine is not None


class TestDistributedLauncher:
    """Test DistributedLauncher functionality."""

    def test_distributed_launcher_exists(self):
        """Test DistributedLauncher exists."""
        from training.distributed import DistributedLauncher
        assert DistributedLauncher is not None


class TestDistributedStrategy:
    """Test DistributedStrategy functionality."""

    def test_distributed_strategy_exists(self):
        """Test DistributedStrategy exists."""
        from training.distributed import ParallelStrategy
        assert ParallelStrategy is not None