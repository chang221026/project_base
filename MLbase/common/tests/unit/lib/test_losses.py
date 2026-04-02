"""Unit tests for LOSSES registry.

Migrated from tests/ut/test_lib_loss_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestLossesRegistry:
    """Test LOSSES registry functionality."""

    def test_losses_registry_exists(self):
        """Test LOSSES registry exists."""
        from lib.loss_func import LOSSES
        assert LOSSES is not None

    def test_base_loss_exists(self):
        """Test BaseLoss exists."""
        from lib.loss_func.base import BaseLoss
        assert BaseLoss is not None

    def test_cross_entropy_registration(self):
        """Test CrossEntropyLoss is registered."""
        from lib.loss_func import LOSSES

        ce = LOSSES.get('CrossEntropyLoss')
        assert ce is not None

    def test_build_cross_entropy(self):
        """Test building CrossEntropyLoss from config."""
        from lib.loss_func import LOSSES

        loss = LOSSES.build({'type': 'CrossEntropyLoss'})
        assert loss is not None

    def test_cross_entropy_compute(self):
        """Test CrossEntropyLoss compute."""
        from lib.loss_func import LOSSES

        loss = LOSSES.build({'type': 'CrossEntropyLoss'})

        import torch
        predictions = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss_value = loss.compute(predictions, targets)
        assert loss_value is not None


class TestBaseLoss:
    """Test BaseLoss functionality."""

    def test_base_loss_compute(self):
        """Test BaseLoss compute method."""
        from lib.loss_func.base import BaseLoss

        class TestLoss(BaseLoss):
            def compute(self, predictions, targets):
                return 0.5

        loss = TestLoss()
        result = loss.compute(None, None)
        assert result == 0.5