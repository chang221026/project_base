"""Unit tests for OPTIMIZERS registry.

Migrated from tests/ut/test_lib_optimizer_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestOptimizersRegistry:
    """Test OPTIMIZERS registry functionality."""

    def test_optimizers_registry_exists(self):
        """Test OPTIMIZERS registry exists."""
        from lib.optimizer import OPTIMIZERS
        assert OPTIMIZERS is not None

    def test_base_optimizer_exists(self):
        """Test BaseOptimizer exists."""
        from lib.optimizer.base import BaseOptimizer
        assert BaseOptimizer is not None

    def test_adam_registration(self):
        """Test Adam is registered."""
        from lib.optimizer import OPTIMIZERS

        adam = OPTIMIZERS.get('Adam')
        assert adam is not None

    def test_build_adam(self):
        """Test building Adam from config."""
        from lib.optimizer import OPTIMIZERS

        import torch
        model = torch.nn.Linear(10, 2)
        params = model.parameters()

        optimizer = OPTIMIZERS.build({
            'type': 'Adam',
            'parameters': list(params),
            'lr': 0.001
        })
        assert optimizer is not None