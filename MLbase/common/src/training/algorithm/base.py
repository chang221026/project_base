"""Base algorithm template.
 
Provides the foundation for all learning algorithm implementations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
 
from utils.logger import get_logger
from utils.config_management import Config
from training.hook.base import BaseHook
 
if TYPE_CHECKING:
    from training.data.dataset_building import SimpleDataLoader
 
 
logger = get_logger()
 
 
def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from state dict keys.
 
    This handles checkpoints saved from DDP-wrapped models,
    allowing them to be loaded into non-DDP models.
 
    Args:
        state_dict: State dictionary with potential 'module.' prefixes.
 
    Returns:
        State dictionary with prefixes removed.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state[k] = v
    return new_state
 
 
class BaseAlgorithm(ABC):
    """Base class for all learning algorithms.
 
    Implements template method pattern for training workflows.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize algorithm.
 
        Args:
            config: Algorithm configuration.
        """
        self.config = Config.from_dict(config or {})
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.evaluator = None
        self.hooks: List[tuple] = []  # List of (priority, BaseHook) tuples
        self._trained = False
        self._current_epoch = 0
        self._global_step = 0
 
    @abstractmethod
    def setup(self) -> None:
        """Setup model, optimizer, loss function, and evaluator."""
        pass
 
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
 
        Args:
            batch: Training batch.
 
        Returns:
            Dictionary of metrics.
        """
        pass
 
    @abstractmethod
    def val_step(self, batch: Any) -> Dict[str, float]:
        """Single validation step.
 
        Args:
            batch: Validation batch.
 
        Returns:
            Dictionary of metrics.
        """
        pass
 
    def train_epoch(self, train_loader: 'SimpleDataLoader') -> Dict[str, float]:
        """Train one epoch.
 
        Args:
            train_loader: Training data loader.
 
        Returns:
            Epoch metrics.
        """
        self.on_epoch_start()
 
        epoch_metrics = {}
        num_batches = len(train_loader)
 
        for batch_idx, batch in enumerate(train_loader):
            self.on_batch_start(batch_idx)
 
            metrics = self.train_step(batch)
 
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
 
            self._global_step += 1
            self.on_batch_end(batch_idx, metrics)
 
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
 
        self.on_epoch_end(epoch_metrics)
        self._current_epoch += 1
 
        return epoch_metrics
 
    def validate(self, val_loader: 'SimpleDataLoader') -> Dict[str, float]:
        """Validate model.
 
        Args:
            val_loader: Validation data loader.
 
        Returns:
            Validation metrics.
        """
        val_metrics = {}
        num_batches = len(val_loader)
 
        for batch in val_loader:
            metrics = self.val_step(batch)
 
            for key, value in metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value
 
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
 
        return val_metrics
 
    def fit(self, train_loader, val_loader=None, epochs: int = 10) -> Dict[str, Any]:
        """Train model.
 
        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of epochs.
 
        Returns:
            Training history.
        """
        logger.info(f"Starting training for {epochs} epochs")
 
        self.on_train_start()
        history = {'train': [], 'val': []}
 
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train'].append(train_metrics)
 
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val'].append(val_metrics)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train: {train_metrics}, Val: {val_metrics}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train: {train_metrics}"
                )
 
        self._trained = True
        self.on_train_end(history)
 
        return history
 
    def predict(self, inputs: Any) -> Any:
        """Make predictions.
 
        Args:
            inputs: Model inputs.
 
        Returns:
            Predictions.
        """
        if not self._trained:
            logger.warning("Model has not been trained yet")
        return self.model(inputs)
 
    def add_hook(self, hook: BaseHook, priority: int = 0) -> None:
        """Add training hook.
 
        Args:
            hook: Hook instance.
            priority: Hook priority (lower = earlier).
        """
        self.hooks.append((priority, hook))
        self.hooks.sort(key=lambda x: x[0])
 
    def on_train_start(self) -> None:
        """Called at the start of training."""
        for _, hook in self.hooks:
            hook.on_train_start(self)
 
    def on_train_end(self, history: Dict[str, Any]) -> None:
        """Called at the end of training."""
        for _, hook in self.hooks:
            hook.on_train_end(self, history)
 
    def on_epoch_start(self) -> None:
        """Called at the start of each epoch."""
        for _, hook in self.hooks:
            hook.on_epoch_start(self)
 
    def on_epoch_end(self, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        for _, hook in self.hooks:
            hook.on_epoch_end(self, metrics)
 
    def on_batch_start(self, batch_idx: int) -> None:
        """Called at the start of each batch."""
        for _, hook in self.hooks:
            hook.on_batch_start(self, batch_idx)
 
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each batch."""
        for _, hook in self.hooks:
            hook.on_batch_end(self, batch_idx, metrics)
 
    def save(self, filepath: str) -> None:
        """Save algorithm state.
 
        Default implementation saves model and optimizer state.
        Subclasses can override for custom state.
 
        Args:
            filepath: Path to save state.
        """
        import torch
 
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': self._current_epoch,
            'global_step': self._global_step,
        }
        torch.save(state, filepath)
        logger.info(f"Algorithm state saved to {filepath}")
 
    def load(self, filepath: str) -> None:
        """Load algorithm state.
 
        Default implementation loads model and optimizer state.
        Handles DDP-wrapped model checkpoints and device mapping.
 
        Args:
            filepath: Path to load state from.
        """
        import torch
        import pickle
 
        device_str = str(self._get_device())
 
        # Try torch.load first, then pickle
        try:
            state = torch.load(filepath, weights_only=False, map_location=device_str)
        except (RuntimeError, pickle.UnpicklingError):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
 
        # Handle different checkpoint formats
        if 'model_state' in state:
            if state['model_state']:
                model_state = _strip_module_prefix(state['model_state'])
                self.model.load_state_dict(model_state)
            if state.get('optimizer_state'):
                self.optimizer.load_state_dict(state['optimizer_state'])
        elif 'model_state_dict' in state:
            model_state = _strip_module_prefix(state['model_state_dict'])
            self.model.load_state_dict(model_state)
            if 'optimizer_state_dict' in state:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
 
        if 'epoch' in state:
            self._current_epoch = state['epoch']
        if 'global_step' in state:
            self._global_step = state['global_step']
 
        logger.info(f"Algorithm state loaded from {filepath}")
 
    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.to_dict()
 
    @property
    def current_epoch(self) -> int:
        """Current training epoch."""
        return self._current_epoch
 
    @property
    def global_step(self) -> int:
        """Global training step."""
        return self._global_step
 
    @property
    def is_trained(self) -> bool:
        """Whether model has been trained."""
        return self._trained
 
    def _parse_batch(self, batch: Any) -> Any:
        """Extract inputs from batch.
 
        Handles common batch formats: (inputs, targets) tuples or plain inputs.
 
        Args:
            batch: Input batch which may be a tuple/list of (inputs, targets)
                   or just inputs.
 
        Returns:
            Extracted inputs.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0]
        return batch
 
    # === 组件构建辅助方法 ===
 
    def _build_component(self, registry, config: Optional[Dict], **kwargs) -> Any:
        """通用组件构建方法。
 
        Args:
            registry: 组件注册表 (如 MODELS, OPTIMIZERS, LOSSES)。
            config: 组件配置字典。
            **kwargs: 额外参数，会合并到配置中。
 
        Returns:
            构建好的组件实例，如果 config 为空则返回 None。
        """
        if not config:
            return None
        full_config = {**config, **kwargs}
        return registry.build(full_config)
 
    def _build_model(self, config: Optional[Dict]) -> Any:
        """构建模型。
 
        Args:
            config: 模型配置字典。
 
        Returns:
            模型实例。
        """
        from lib.models import MODELS
        return self._build_component(MODELS, config)
 
    def _build_optimizer(self, model_or_params, config: Optional[Dict] = None) -> Any:
        """构建优化器。
 
        Args:
            model_or_params: 模型实例或参数迭代器。
            config: 优化器配置字典。
 
        Returns:
            优化器实例。
        """
        from lib.optimizer import OPTIMIZERS
        config = config or {'type': 'Adam', 'lr': 0.001}
        # 获取参数
        if hasattr(model_or_params, 'parameters'):
            params = model_or_params.parameters()
        else:
            params = model_or_params
        config['parameters'] = params
        return OPTIMIZERS.build(config)
 
    def _build_loss(self, config: Optional[Dict] = None) -> Any:
        """构建损失函数。
 
        Args:
            config: 损失函数配置字典。
 
        Returns:
            损失函数实例。
        """
        from lib.loss_func import LOSSES
        config = config or {'type': 'CrossEntropyLoss'}
        return LOSSES.build(config)
 
    def _build_evaluator(self, config: Optional[Dict] = None) -> Any:
        """构建评估器。
 
        Args:
            config: 评估器配置字典。
 
        Returns:
            评估器实例。
        """
        from lib.evaluator import EVALUATORS
        config = config or {'type': 'AccuracyEvaluator'}
        return EVALUATORS.build(config)
 
    # === 设备管理 ===
 
    def _get_device(self):
        """获取当前设备。
 
        Returns:
            设备对象。
        """
        from utils.device_management import get_device_manager, Device, DeviceType
        device = get_device_manager().get_current_device()
        if device is None:
            # Fallback to CPU if device is None
            return Device(DeviceType.CPU, 0)
        return device
 
    def _move_to_device(self, data: Any, device=None, keep_dtype: bool = False) -> Any:
        """将数据移动到设备。
 
        处理张量、numpy 数组和嵌套结构。
        对于数值数据默认转换为 float32，除非 keep_dtype=True。
 
        Args:
            data: 要移动的数据（张量、numpy 数组、列表、元组或字典）。
            device: 目标设备，如果为 None 则使用当前设备。
            keep_dtype: 如果为 True，保持原始 dtype 而不是转换为 float32。
 
        Returns:
            在目标设备上的数据。
        """
        import torch
        import numpy as np
 
        if device is None:
            device = self._get_device()
        device_str = str(device)
 
        if isinstance(data, torch.Tensor):
            tensor = data.to(device_str)
            if not keep_dtype and tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.float()
            return tensor
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(device_str)
            if not keep_dtype and tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.float()
            return tensor
        elif isinstance(data, dict):
            return {k: self._move_to_device(v, device, keep_dtype) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            moved = [self._move_to_device(item, device, keep_dtype) for item in data]
            if all(isinstance(item, torch.Tensor) for item in moved):
                return torch.stack(moved)
            try:
                if keep_dtype:
                    return torch.tensor(moved, device=device_str)
                else:
                    return torch.tensor(moved, dtype=torch.float32, device=device_str)
            except Exception:
                return type(data)(moved)
        elif isinstance(data, (int, float)):
            if keep_dtype:
                return torch.tensor(data, device=device_str)
            else:
                return torch.tensor(data, dtype=torch.float32, device=device_str)
        else:
            try:
                if keep_dtype:
                    return torch.tensor(data, device=device_str)
                else:
                    return torch.tensor(data, dtype=torch.float32, device=device_str)
            except Exception:
                return data
 
 
class eval_mode:
    """Context manager for evaluation mode.
 
    Temporarily sets a model to evaluation mode and restores
    the original training state on exit.
    """
 
    def __init__(self, model):
        """Initialize context manager.
 
        Args:
            model: Model to set to eval mode.
        """
        self.model = model
        self.training = None
 
    def __enter__(self) -> 'eval_mode':
        """Enter evaluation mode context.
 
        Returns:
            Self for use in with statements.
        """
        if hasattr(self.model, 'training'):
            self.training = self.model.training
            if hasattr(self.model, 'eval'):
                self.model.eval()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit evaluation mode context and restore training state."""
        if self.training is not None and hasattr(self.model, 'train'):
            if self.training:
                self.model.train()