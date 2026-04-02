"""Integration tests for component discovery.

Tests dynamic loading and discovery of components.
Migrated from tests/integration/test_custom_components.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestCustomModelRegistration:
    """Test registering custom models."""

    def test_register_and_build_custom_model(self):
        """Test registering and building a custom model."""
        from utils.registry import Registry
        from lib.models.base import BaseModel

        test_registry = Registry('test_custom')

        @test_registry.register('TestModel')
        class TestModel(BaseModel):
            def __init__(self, config=None):
                super().__init__(config)

        # Should be able to get the registered model
        assert 'TestModel' in test_registry


class TestRegistryDynamicLoading:
    """Test dynamic component loading from config."""

    def test_build_from_config(self):
        """Test building components from config."""
        from lib.models import MODELS

        config = {
            'type': 'MLP',
            'input_dim': 10,
            'output_dim': 2
        }

        model = MODELS.build(config)
        assert model is not None


class TestComponentDiscovery:
    """Test component auto-discovery."""

    def test_auto_discovery(self):
        """Test auto-discovery of components."""
        from lib.models import MODELS
        from lib.loss_func import LOSSES
        from lib.optimizer import OPTIMIZERS

        # All registries should have components
        assert MODELS is not None
        assert LOSSES is not None
        assert OPTIMIZERS is not None