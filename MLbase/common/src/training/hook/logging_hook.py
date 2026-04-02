"""Logging hook for training metrics."""
from typing import Any, Dict, Optional
import time
 
from .base import BaseHook
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class LoggingHook(BaseHook):
    """Hook for logging training progress and metrics."""
 
    def __init__(self,
                 log_interval: int = 10,
                 log_every_n_epochs: int = 1,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 0):
        """Initialize logging hook.
 
        Args:
            log_interval: Log every N batches.
            log_every_n_epochs: Log every N epochs.
            config: Additional configuration.
            priority: Hook priority.
        """
        super().__init__(config, priority)
        self.log_interval = log_interval
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_start_time = 0
        self.train_start_time = 0
 
    def on_train_start(self, trainer) -> None:
        """Log training start."""
        self.train_start_time = time.time()
        logger.info("=" * 50)
        logger.info("Training started")
        logger.info(f"Config: {trainer.get_config()}")
        logger.info("=" * 50)
 
    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """Log training end."""
        elapsed = time.time() - self.train_start_time
        logger.info("=" * 50)
        logger.info(f"Training completed in {elapsed:.2f}s")
 
        # Log final metrics
        if history.get('train'):
            final_train = history['train'][-1]
            logger.info(f"Final train metrics: {final_train}")
        if history.get('val'):
            final_val = history['val'][-1]
            logger.info(f"Final val metrics: {final_val}")
        logger.info("=" * 50)
 
    def on_epoch_start(self, trainer) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log epoch metrics."""
        epoch = trainer.current_epoch
 
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
 
        elapsed = time.time() - self.epoch_start_time
        global_step = trainer.global_step
 
        # Format metrics
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
 
        logger.info(
            f"Epoch {epoch + 1} | Step {global_step} | "
            f"Time {elapsed:.2f}s | {metrics_str}"
        )
 
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict[str, float]) -> None:
        """Log batch metrics."""
        if (batch_idx + 1) % self.log_interval != 0:
            return
 
        epoch = trainer.current_epoch
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
 
        logger.debug(
            f"Epoch {epoch + 1} | Batch {batch_idx + 1} | {metrics_str}"
        )
 
 
class TensorBoardHook(BaseHook):
    """Hook for TensorBoard logging."""
 
    def __init__(self,
                 log_dir: str = './runs',
                 log_every_n_steps: int = 10,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 0):
        """Initialize TensorBoard hook.
 
        Args:
            log_dir: TensorBoard log directory.
            log_every_n_steps: Log every N steps.
            config: Additional configuration.
            priority: Hook priority.
        """
        super().__init__(config, priority)
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.writer = None
 
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log epoch metrics to TensorBoard."""
        if not self.writer:
            return
 
        epoch = trainer.current_epoch
 
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", value, epoch)
 
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict[str, float]) -> None:
        """Log batch metrics to TensorBoard."""
        if not self.writer:
            return
 
        if (batch_idx + 1) % self.log_every_n_steps != 0:
            return
 
        global_step = trainer.global_step
 
        for name, value in metrics.items():
            self.writer.add_scalar(f"batch/{name}", value, global_step)
 
    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard writer closed")