"""Learning rate scheduler hook."""
from typing import Any, Dict, Optional, Union
 
from .base import BaseHook
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class LRSchedulerHook(BaseHook):
    """Hook for learning rate scheduling.
 
    Adjusts learning rate based on epochs or metrics.
    """
 
    def __init__(self,
                 scheduler_type: str = 'step',
                 step_size: int = 30,
                 gamma: float = 0.1,
                 warmup_epochs: int = 0,
                 warmup_lr: float = 1e-4,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 5):
        """Initialize LR scheduler hook.
 
        Args:
            scheduler_type: Type of scheduler ('step', 'exponential', 'plateau').
            step_size: Epochs between LR updates.
            gamma: LR decay factor.
            warmup_epochs: Number of warmup epochs.
            warmup_lr: Initial warmup learning rate.
            config: Additional configuration.
            priority: Hook priority.
        """
        super().__init__(config, priority)
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = None
 
    def on_train_start(self, trainer) -> None:
        """Initialize scheduler."""
        if trainer.optimizer:
            self.base_lr = trainer.optimizer.get_lr() or self.warmup_lr
            logger.info(f"LR scheduler initialized: {self.scheduler_type}")
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Update learning rate."""
        if not trainer.optimizer:
            return
 
        epoch = trainer.current_epoch
        current_lr = trainer.optimizer.get_lr()
 
        # Warmup
        if epoch < self.warmup_epochs:
            new_lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
            trainer.optimizer.set_lr(new_lr)
            logger.debug(f"Warmup LR: {new_lr:.6f}")
            return
 
        # Regular scheduling
        if self.scheduler_type == 'step':
            if (epoch - self.warmup_epochs + 1) % self.step_size == 0:
                new_lr = current_lr * self.gamma
                trainer.optimizer.set_lr(new_lr)
                logger.info(f"LR updated: {current_lr:.6f} -> {new_lr:.6f}")
 
        elif self.scheduler_type == 'exponential':
            new_lr = self.base_lr * (self.gamma ** epoch)
            trainer.optimizer.set_lr(new_lr)
            logger.debug(f"LR: {new_lr:.6f}")
 
        elif self.scheduler_type == 'plateau':
            # Requires monitoring a metric
            monitor = self.config.get('monitor', 'val_loss')
            if monitor in metrics:
                # Simple implementation - could be more sophisticated
                pass
 
    def get_lr(self, trainer) -> float:
        """Get current learning rate."""
        if trainer.optimizer:
            return trainer.optimizer.get_lr()
        return 0.0