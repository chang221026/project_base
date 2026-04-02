"""pytest configuration and fixtures for ML framework tests."""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import base classes for concrete implementations
from lib.models.base import BaseModel
from lib.optimizer.base import BaseOptimizer
from lib.evaluator.base import BaseEvaluator
from lib.loss_func.base import BaseLoss
from training.hook.base import BaseHook
from training.algorithm.base import BaseAlgorithm


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def temp_project_dir(temp_dir: Path) -> Path:
    """Alias for temp_dir for integration tests.

    This fixture exists for backwards compatibility with integration tests
    that expect a temp_project_dir fixture.
    """
    return temp_dir


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for tests."""
    file_path = temp_dir / "test_file.txt"
    file_path.touch()
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def temp_yaml_file(temp_dir: Path) -> Path:
    """Create a temporary YAML config file."""
    import yaml
    config = {
        'model': {'type': 'test_model', 'hidden_dim': 128},
        'training': {'epochs': 10, 'batch_size': 32},
        'optimizer': {'type': 'adam', 'lr': 0.001}
    }
    file_path = temp_dir / "config.yaml"
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def temp_json_file(temp_dir: Path) -> Path:
    """Create a temporary JSON config file."""
    import json
    config = {
        'model': {'type': 'test_model', 'hidden_dim': 128},
        'training': {'epochs': 10, 'batch_size': 32}
    }
    file_path = temp_dir / "config.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return file_path


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_torch():
    """Mock PyTorch module."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.cuda.device_count.return_value = 0
    return mock


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.training = True
    model.state_dict.return_value = {'weight': MagicMock()}
    model.eval.return_value = model
    model.train.return_value = model
    model.parameters.return_value = [MagicMock()]
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {'state': {}, 'lr': 0.001}
    optimizer.load_state_dict.return_value = None
    optimizer.get_lr.return_value = 0.001
    optimizer.set_lr.return_value = None
    return optimizer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = MagicMock()
    dataloader.__len__ = lambda self: 10
    dataloader.__iter__ = lambda self: iter([
        (MagicMock(), MagicMock()) for _ in range(10)
    ])
    return dataloader


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        'model': {
            'type': 'TestModel',
            'hidden_dim': 128,
            'num_layers': 3
        },
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'optimizer': {
            'type': 'Adam',
            'lr': 0.001,
            'weight_decay': 0.0001
        }
    }


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        'type': 'TestModel',
        'hidden_dim': 128,
        'num_layers': 3
    }


# ============================================================================
# Singleton Reset Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test.

    By default, hide all accelerators so tests run in CPU-only mode.
    Use 'multi_npu_env' fixture for tests that need to see real NPU devices.
    """
    # Reset Logger singleton
    from utils.logger import Logger
    Logger._instance = None
    Logger._initialized = False

    # Reset DeviceManager singleton
    from utils.device_management import DeviceManager
    DeviceManager._instance = None
    DeviceManager._initialized = False

    # Reset DistributedManager singleton
    from utils.distributed_comm import DistributedManager
    DistributedManager._instance = None
    DistributedManager._initialized = False

    # Hide all accelerators by default (CPU-only mode for tests)
    os.environ['ASCEND_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    yield

    # Clean up after test
    Logger._instance = None
    Logger._initialized = False
    DeviceManager._instance = None
    DeviceManager._initialized = False
    DistributedManager._instance = None
    DistributedManager._initialized = False


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    env_keys = [
        'CUDA_VISIBLE_DEVICES',
        'ASCEND_VISIBLE_DEVICES',
        'MASTER_ADDR', 'MASTER_PORT',
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
        'ML_FRAMEWORK_LAUNCHED',
        'ML_FRAMEWORK_RESULT_FILE'
    ]
    original = {}
    for key in env_keys:
        original[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    yield

    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def multi_npu_env(clean_env):
    """Enable detection of real NPU devices for multi-NPU tests.

    Use this fixture for tests that need to test NPU-related behavior.
    Tests using this fixture will see real NPU devices if available.
    If only CPU is available, tests will still pass.
    """
    # Remove the hiding of accelerators
    os.environ.pop('ASCEND_VISIBLE_DEVICES', None)
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    # Reset singleton so it can detect devices again
    from utils.device_management import DeviceManager
    DeviceManager._instance = None
    DeviceManager._initialized = False

    yield

    # Cleanup - hide accelerators again
    os.environ['ASCEND_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


@pytest.fixture
def env_with_ml_prefix():
    """Set environment variables with ML_ prefix."""
    original = {}
    test_vars = {
        'ML_TRAINING_EPOCHS': '20',
        'ML_TRAINING_BATCH_SIZE': '64',
        'ML_MODEL_HIDDEN_DIM': '256',
        'ML_DEBUG_MODE': 'true'
    }
    for key, value in test_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        else:
            del os.environ[key]


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_registry():
    """Create a test registry with sample components."""
    from utils.registry import Registry

    registry = Registry('test')

    @registry.register('SimpleModel')
    class SimpleModel:
        def __init__(self, dim=10):
            self.dim = dim

    @registry.register('ComplexModel')
    class ComplexModel:
        def __init__(self, hidden_dim=64, num_layers=2):
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

    return registry


def create_test_checkpoint(temp_dir: Path, epoch: int = 1) -> Path:
    """Create a test checkpoint file."""
    import pickle
    checkpoint = {
        'epoch': epoch,
        'model_state': {'weight': f'data_{epoch}'},
        'optimizer_state': {'lr': 0.001},
        'metrics': {'loss': 0.5}
    }
    checkpoint_path = temp_dir / f"checkpoint_epoch_{epoch}.pth"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    return checkpoint_path


# ============================================================================
# Algorithm Fixtures
# ============================================================================

@pytest.fixture
def simple_algorithm():
    """Create a simple test algorithm for integration tests."""
    from training.algorithm.base import BaseAlgorithm
    from unittest.mock import MagicMock

    class SimpleTestAlgorithm(BaseAlgorithm):
        def __init__(self, config=None):
            super().__init__(config)
            self.should_stop = False

        def setup(self):
            self.model = MagicMock()
            self.model.state_dict.return_value = {'weight': MagicMock()}
            self.model.load_state_dict = MagicMock()
            self.optimizer = MagicMock()
            self.optimizer.state_dict.return_value = {'state': {}, 'lr': 0.001}
            self.optimizer.load_state_dict = MagicMock()
            self.optimizer.get_lr.return_value = 0.001
            self.optimizer.set_lr = MagicMock()
            self.loss_fn = MagicMock()

        def train_step(self, batch):
            return {'loss': 0.5, 'accuracy': 0.8}

        def val_step(self, batch):
            return {'val_loss': 0.4, 'val_accuracy': 0.85}

    return SimpleTestAlgorithm


@pytest.fixture
def simple_train_dataloader():
    """Create a simple train dataloader for testing."""
    mock_loader = MagicMock()
    mock_loader.__len__ = lambda self: 10
    mock_loader.__iter__ = lambda self: iter([
        (MagicMock(), MagicMock()) for _ in range(10)
    ])
    return mock_loader


@pytest.fixture
def simple_val_dataloader():
    """Create a simple validation dataloader for testing."""
    mock_loader = MagicMock()
    mock_loader.__len__ = lambda self: 5
    mock_loader.__iter__ = lambda self: iter([
        (MagicMock(), MagicMock()) for _ in range(5)
    ])
    return mock_loader


# ============================================================================
# Registry Fixtures
# ============================================================================

@pytest.fixture
def model_registry():
    """Create a model registry with test models."""
    from utils.registry import Registry
    from lib.models.base import BaseModel

    registry = Registry('test_models')

    @registry.register('SimpleMLP')
    class SimpleMLP(BaseModel):
        def __init__(self, hidden_dim=64, num_layers=2, config=None):
            super().__init__(config)
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self._built = False

        def forward(self, x):
            return x

        def build(self, input_shape=None):
            self._built = True
            return self

    @registry.register('ComplexModel')
    class ComplexModel(BaseModel):
        def __init__(self, hidden_dim=128, dropout=0.1, config=None):
            super().__init__(config)
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self._built = False

        def forward(self, x):
            return x

        def build(self, input_shape=None):
            self._built = True
            return self

    return registry


@pytest.fixture
def loss_registry():
    """Create a loss function registry with test losses."""
    from utils.registry import Registry
    from lib.loss_func.base import BaseLoss

    registry = Registry('test_losses')

    @registry.register('MSELoss')
    class MSELoss(BaseLoss):
        def compute(self, pred, target):
            return 0.5

    @registry.register('CrossEntropyLoss')
    class CrossEntropyLoss(BaseLoss):
        def compute(self, pred, target):
            return 0.3

    return registry


@pytest.fixture
def optimizer_registry():
    """Create an optimizer registry with test optimizers."""
    from utils.registry import Registry
    from lib.optimizer.base import BaseOptimizer

    registry = Registry('test_optimizers')

    @registry.register('SGD')
    class SGD(BaseOptimizer):
        def step(self):
            pass

        def zero_grad(self):
            pass

    @registry.register('Adam')
    class Adam(BaseOptimizer):
        def step(self):
            pass

        def zero_grad(self):
            pass

    return registry


@pytest.fixture
def evaluator_registry():
    """Create an evaluator registry with test evaluators."""
    from utils.registry import Registry
    from lib.evaluator.base import BaseEvaluator

    registry = Registry('test_evaluators')

    @registry.register('AccuracyEvaluator')
    class AccuracyEvaluator(BaseEvaluator):
        def evaluate(self, predictions, targets):
            return {'accuracy': 0.85}

        def compute_metrics(self):
            if not self.metrics.get('accuracy'):
                return {'accuracy': 0.0}
            return {'accuracy': sum(self.metrics['accuracy']) / len(self.metrics['accuracy'])}

    @registry.register('MultiMetricEvaluator')
    class MultiMetricEvaluator(BaseEvaluator):
        def evaluate(self, predictions, targets):
            return {'accuracy': 0.85, 'f1': 0.80, 'precision': 0.82}

        def compute_metrics(self):
            result = {}
            for key, values in self.metrics.items():
                if values:
                    result[key] = sum(values) / len(values)
            return result

    return registry


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import numpy as np
    np.random.seed(42)
    data = np.random.randn(100, 10).tolist()
    targets = np.random.randint(0, 2, 100).tolist()
    return data, targets


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    from training.data.dataset_building import Dataset

    data = list(range(100))
    targets = [i % 2 for i in range(100)]

    return Dataset(data, targets)


@pytest.fixture
def sample_dataloader(sample_dataset):
    """Create a sample dataloader for testing."""
    from training.data.dataset_building import DataLoader
    return DataLoader(sample_dataset, batch_size=10, shuffle=True)


# ============================================================================
# Concrete Implementations for Testing Abstract Base Classes
# ============================================================================

class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing.

    Simple model that doubles inputs in forward pass.
    """

    def forward(self, inputs):
        """Simple forward pass that doubles inputs."""
        return inputs * 2

    def build(self, input_shape=None):
        """Build the model."""
        self._built = True
        return self


class ConcreteOptimizer(BaseOptimizer):
    """Concrete implementation of BaseOptimizer for testing.

    Simple optimizer that tracks step count.
    """

    def step(self):
        """Perform one optimization step."""
        self._step_count += 1

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if hasattr(param, 'grad'):
                param.grad = None


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing.

    Simple evaluator that computes accuracy.
    """

    def evaluate(self, predictions, targets):
        """Simple evaluation returning accuracy."""
        correct = sum(p == t for p, t in zip(predictions, targets))
        total = len(predictions)
        return {'accuracy': correct / total if total > 0 else 0.0}

    def compute_metrics(self):
        """Compute average metrics."""
        if not self.metrics:
            return {}
        return {
            key: sum(values) / len(values)
            for key, values in self.metrics.items()
        }


class ConcreteLoss(BaseLoss):
    """Concrete implementation of BaseLoss for testing.

    Simple loss using mean absolute error.
    """

    def compute(self, predictions, targets):
        """Simple loss computation using mean absolute error."""
        total = sum(abs(p - t) for p, t in zip(predictions, targets))
        return total / len(predictions) if predictions else 0.0


class ConcreteHook(BaseHook):
    """Concrete implementation of BaseHook for testing.

    Hook that tracks which callbacks were called.
    """

    def on_train_start(self, trainer):
        """Called at the start of training."""
        self.train_start_called = True

    def on_train_end(self, trainer, history):
        """Called at the end of training."""
        self.train_end_called = True

    def on_epoch_start(self, trainer):
        """Called at the start of each epoch."""
        self.epoch_start_called = True

    def on_epoch_end(self, trainer, metrics):
        """Called at the end of each epoch."""
        self.epoch_end_called = True

    def on_batch_start(self, trainer, batch_idx):
        """Called at the start of each batch."""
        self.batch_start_called = True

    def on_batch_end(self, trainer, batch_idx, metrics):
        """Called at the end of each batch."""
        self.batch_end_called = True


class ConcreteAlgorithm(BaseAlgorithm):
    """Concrete implementation of BaseAlgorithm for testing.

    Simple algorithm with mock components.
    """

    def setup(self):
        """Setup mock components."""
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.loss_fn = MagicMock()

    def train_step(self, batch):
        """Mock train step."""
        return {'loss': 0.5}

    def val_step(self, batch):
        """Mock val step."""
        return {'loss': 0.4, 'accuracy': 0.9}