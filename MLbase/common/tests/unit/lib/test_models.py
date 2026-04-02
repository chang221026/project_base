"""Unit tests for MODELS registry.

Migrated from tests/ut/test_lib_models_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestModelsRegistry:
    """Test MODELS registry functionality."""

    def test_models_registry_exists(self):
        """Test MODELS registry exists."""
        from lib.models import MODELS
        assert MODELS is not None

    def test_base_model_exists(self):
        """Test BaseModel exists."""
        from lib.models.base import BaseModel
        assert BaseModel is not None

    def test_mlp_registration(self):
        """Test MLP is registered."""
        from lib.models import MODELS

        mlp = MODELS.get('MLP')
        assert mlp is not None

    def test_build_mlp(self):
        """Test building MLP from config."""
        from lib.models import MODELS

        model = MODELS.build({
            'type': 'MLP',
            'input_dim': 10,
            'output_dim': 2
        })
        assert model is not None

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from lib.models import MODELS

        model = MODELS.build({
            'type': 'MLP',
            'input_dim': 10,
            'output_dim': 2
        })

        # Test forward
        import torch
        x = torch.randn(4, 10)
        y = model(x)
        assert y.shape == (4, 2)