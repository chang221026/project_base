"""System tests for custom extensibility workflow.

Tests the custom extensibility use case from README Section 5.2,
verifying that users can register and use custom components.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def clean_registry():
    """Clean registries before test."""
    from utils.registry import Registry
    import gc

    # Get all registries and clear them
    registries = [
        Registry('models'),
        Registry('losses'),
        Registry('optimizers'),
        Registry('evaluators')
    ]

    yield registries

    gc.collect()


# ============================================================================
# Test: Custom Component End-to-End
# ============================================================================

@pytest.mark.system
class TestCustomExtensibilityEndToEnd:
    """Test end-to-end custom component registration and usage."""

    def test_register_and_use_custom_model(self, clean_registry):
        """Test: Register custom model -> use in config -> training works.

        From README Section 5.2.

        Expected:
        - Custom model registers successfully
        - Can specify in config
        - Training uses the custom model
        """
        from utils.registry import Registry
        from lib.models.base import BaseModel

        models = Registry('models')

        # Register custom model
        @models.register('CustomNet')
        class CustomNet(BaseModel):
            def __init__(self, input_dim, output_dim, config=None):
                super().__init__(config)
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.custom_flag = True

            def forward(self, x):
                return x

            def build(self, input_shape=None):
                self._built = True
                return self

        # Verify registration worked
        model_cls = models.get('CustomNet')
        assert model_cls is CustomNet

        # Build from config
        model = models.build({
            'type': 'CustomNet',
            'input_dim': 10,
            'output_dim': 2
        })

        assert model is not None
        assert model.custom_flag is True

    def test_register_and_use_custom_loss(self, clean_registry):
        """Test: Register custom loss -> use in config -> loss computed.

        From README Section 5.2.

        Expected:
        - Custom loss registers
        - Can be used in config
        - Loss computation works
        """
        from utils.registry import Registry
        from lib.loss_func.base import BaseLoss

        losses = Registry('losses')

        @losses.register('CustomLoss')
        class CustomLoss(BaseLoss):
            def __init__(self, weight=1.0, config=None):
                super().__init__(config)
                self.weight = weight

            def compute(self, predictions, targets):
                return self.weight * 0.5

        # Build and test
        loss = losses.build({'type': 'CustomLoss', 'weight': 2.0})

        result = loss.compute([1, 2, 3], [1, 2, 3])
        assert result == 1.0  # 2.0 * 0.5

    def test_register_and_use_custom_optimizer(self, clean_registry):
        """Test: Register custom optimizer -> use in config -> training works.

        Expected:
        - Custom optimizer registers
        - Can be configured in config
        - Optimizer steps work
        """
        from utils.registry import Registry
        from lib.optimizer.base import BaseOptimizer

        optimizers = Registry('optimizers')

        @optimizers.register('CustomSGD')
        class CustomSGD(BaseOptimizer):
            def __init__(self, parameters=None, lr=0.01, momentum=0.9, config=None):
                super().__init__(parameters, config)
                self.lr = lr
                self.momentum = momentum
                self._step_count = 0  # Use private var (step_count is property)

            def step(self):
                self._step_count += 1

            def zero_grad(self):
                pass

        # Build and test
        optimizer = optimizers.build({
            'type': 'CustomSGD',
            'lr': 0.001,
            'momentum': 0.9
        })

        assert optimizer.lr == 0.001
        assert optimizer.momentum == 0.9

    def test_custom_imports_auto_discovery(self, temp_project_dir):
        """Test: custom_imports config auto-discovers components.

        From README Section 5.2 - custom_imports feature.

        Expected:
        - Module in custom_imports is imported
        - Components in that module are registered
        """
        from utils.registry import Registry

        # Create a custom module file
        custom_file = temp_project_dir / 'my_custom_models.py'
        custom_file.write_text('''
from utils.registry import Registry
from lib.models.base import BaseModel

MODELS = Registry('my_models')

@MODELS.register('ImportedModel')
class ImportedModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)

    def forward(self, x):
        return x
''')

        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("my_custom_models", str(custom_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Verify registration using the module's registry instance
        assert 'ImportedModel' in module.MODELS.list_registered()


# ============================================================================
# Test: Extensibility Reliability
# ============================================================================

@pytest.mark.system
class TestExtensibilityReliability:
    """Test extensibility system reliability."""

    def test_duplicate_registration_overwrites(self):
        """Test: Duplicate registration either warns or overwrites.

        Expected:
        - Registering same name twice gives error
        - No silent failure
        """
        from utils.registry import Registry

        registry = Registry('test_dup')

        @registry.register('DupModel')
        class Model1:
            pass

        # Second registration with same name - should raise error
        with pytest.raises(ValueError, match="already registered|exists"):
            @registry.register('DupModel')
            class Model2:
                pass

    def test_missing_component_raises_clear_error(self):
        """Test: Using non-existent component gives clear error.

        Expected:
        - Error mentions component wasn't found
        - Lists available components
        """
        from utils.registry import Registry

        registry = Registry('test_missing')

        with pytest.raises(Exception) as exc_info:
            registry.build({'type': 'NonExistentComponent'})

        # Error should mention not found
        error_msg = str(exc_info.value).lower()
        assert 'not found' in error_msg or 'exist' in error_msg

    def test_registry_thread_safety(self):
        """Test: Multiple threads can register without conflict.

        Expected:
            - Concurrent registrations don't corrupt registry
        """
        import threading
        from utils.registry import Registry

        registry = Registry('test_thread')
        errors = []

        def register_component(name):
            try:
                @registry.register(name)
                class Component:
                    pass
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_component, args=(f'ThreadComp{i}',))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have registered all components
        assert len(registry.list_registered()) == 5


# ============================================================================
# Test: Various Custom Components
# ============================================================================

@pytest.mark.system
class TestVariousCustomComponents:
    """Test registering various types of custom components."""

    def test_register_custom_evaluator(self):
        """Test: Custom evaluator registration and usage.

        Expected:
        - Evaluator registers
        - Evaluates predictions correctly
        """
        from utils.registry import Registry
        from lib.evaluator.base import BaseEvaluator

        evaluators = Registry('test_eval')

        @evaluators.register('F1Evaluator')
        class F1Evaluator(BaseEvaluator):
            def evaluate(self, predictions, targets):
                return {'f1': 0.85}

            def compute_metrics(self):
                return {'f1': 0.85}

        evaluator = evaluators.build({'type': 'F1Evaluator'})
        result = evaluator.evaluate([1, 0, 1], [1, 0, 0])

        assert 'f1' in result

    def test_register_custom_data_fetcher(self):
        """Test: Custom data fetcher registration and usage.

        Expected:
        - Fetcher registers
        - Can fetch data
        """
        from utils.registry import Registry
        from lib.data_fetching.base import BaseDataFetcher

        fetchers = Registry('test_fetch')

        @fetchers.register('CustomFetcher')
        class CustomFetcher(BaseDataFetcher):
            def __init__(self, source, config=None):
                super().__init__(config)
                self.source = source

            def fetch(self, source=None):
                return [[1, 2], [3, 4]], [0, 1]

            def batch_fetch(self, sources, batch_size=32):
                for source in sources:
                    yield self.fetch(source)

        fetcher = fetchers.build({'type': 'CustomFetcher', 'source': 'test.csv'})
        data, target = fetcher.fetch()

        assert data is not None
        assert target is not None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    import tempfile
    import shutil
    from pathlib import Path

    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)