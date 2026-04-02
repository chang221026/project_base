"""Early stopping hook to stop training when validation stops improving."""
from typing import Any, Dict, Optional
 
from .base import BaseHook
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class EarlyStoppingHook(BaseHook):
    """Hook for early stopping.
 
    Stops training when a monitored metric stops improving.
    """
 
    def __init__(self,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 verbose: bool = True,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 0):
        """Initialize early stopping hook.
 
        Args:
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
            patience: Number of epochs with no improvement after which to stop.
            min_delta: Minimum change to qualify as improvement.
            verbose: Whether to print messages.
            config: Additional configuration.
            priority: Hook priority.
        """
        super().__init__(config, priority)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
 
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Check if training should stop."""
        if self.early_stop:
            return
 
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return
 
        score = metrics[self.monitor]
        current_epoch = trainer.current_epoch
 
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = current_epoch
 
            if self.verbose:
                logger.info(
                    f"Epoch {current_epoch + 1}: {self.monitor} improved to {score:.4f}"
                )
        else:
            self.counter += 1
 
            if self.verbose:
                logger.info(
                    f"Epoch {current_epoch + 1}: {self.monitor} did not improve "
                    f"({self.counter}/{self.patience})"
                )
 
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.should_stop = True  # Signal trainer to stop
 
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {current_epoch + 1}. "
                        f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch + 1}"
                    )
 
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta
 
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
 
    def get_best_score(self) -> float:
        """Get best score seen so far."""
        return self.best_score
 
    def get_best_epoch(self) -> int:
        """Get epoch with best score."""
        return self.best_epoch
 
    def should_stop_early(self) -> bool:
        """Check if early stopping has been triggered."""
        return self.early_stop