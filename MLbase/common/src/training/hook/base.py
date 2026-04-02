"""Base training hook class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
 
 
class BaseHook(ABC):
    """Base class for all training hooks.
 
    Hooks allow custom actions at various points during training.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None, priority: int = 0):
        """Initialize hook.
 
        Args:
            config: Hook configuration.
            priority: Hook priority (lower = earlier execution).
        """
        self.config = config or {}
        self.priority = priority
        self.enabled = True
 
    def on_train_start(self, trainer) -> None:
        """Called at the start of training.
 
        Args:
            trainer: Trainer/algorithm instance.
        """
        pass
 
    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """Called at the end of training.
 
        Args:
            trainer: Trainer/algorithm instance.
            history: Training history.
        """
        pass
 
    def on_epoch_start(self, trainer) -> None:
        """Called at the start of each epoch.
 
        Args:
            trainer: Trainer/algorithm instance.
        """
        pass
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch.
 
        Args:
            trainer: Trainer/algorithm instance.
            metrics: Epoch metrics.
        """
        pass
 
    def on_batch_start(self, trainer, batch_idx: int) -> None:
        """Called at the start of each batch.
 
        Args:
            trainer: Trainer/algorithm instance.
            batch_idx: Batch index.
        """
        pass
 
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each batch.
 
        Args:
            trainer: Trainer/algorithm instance.
            batch_idx: Batch index.
            metrics: Batch metrics.
        """
        pass
 
    def on_validation_start(self, trainer) -> None:
        """Called at the start of validation.
 
        Args:
            trainer: Trainer/algorithm instance.
        """
        pass
 
    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Called at the end of validation.
 
        Args:
            trainer: Trainer/algorithm instance.
            metrics: Validation metrics.
        """
        pass
 
    def enable(self) -> None:
        """Enable hook."""
        self.enabled = True
 
    def disable(self) -> None:
        """Disable hook."""
        self.enabled = False
 
    def get_config(self) -> Dict[str, Any]:
        """Get hook configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(priority={self.priority}, enabled={self.enabled})"