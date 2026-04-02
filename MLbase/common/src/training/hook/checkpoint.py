"""Checkpoint hook for saving model checkpoints."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
 
from .base import BaseHook
from utils.io import ensure_dir, save_pickle
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class CheckpointHook(BaseHook):
    """Hook for saving model checkpoints.
 
    Saves checkpoints at specified intervals or when validation improves.
    """
 
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 save_interval: int = 1,
                 save_best: bool = True,
                 monitor: str = 'loss',
                 mode: str = 'min',
                 max_keep: int = 5,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 10):
        """Initialize checkpoint hook.
 
        Args:
            checkpoint_dir: Directory to save checkpoints.
            save_interval: Save every N epochs.
            save_best: Whether to save best model.
            monitor: Metric to monitor for best model.
            mode: 'min' or 'max' for best model.
            max_keep: Maximum checkpoints to keep.
            config: Additional configuration.
            priority: Hook priority.
        """
        super().__init__(config, priority)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.max_keep = max_keep
 
        ensure_dir(self.checkpoint_dir)
 
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
 
    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Save checkpoint at end of epoch."""
        epoch = trainer.current_epoch
 
        # Regular checkpoint
        if (epoch + 1) % self.save_interval == 0:
            self._save_checkpoint(trainer, epoch, metrics, is_best=False)
 
        # Best checkpoint
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            is_best = self._is_better(score)
 
            if is_best:
                self.best_score = score
                self._save_checkpoint(trainer, epoch, metrics, is_best=True)
                logger.info(
                    f"New best {self.monitor}: {score:.4f} at epoch {epoch + 1}"
                )
 
    def _is_better(self, score: float) -> bool:
        """Check if score is better than best."""
        if self.mode == 'min':
            return score < self.best_score
        return score > self.best_score
 
    def _save_checkpoint(self, trainer, epoch: int,
                         metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save checkpoint.
 
        Args:
            trainer: Trainer instance.
            epoch: Current epoch.
            metrics: Current metrics.
            is_best: Whether this is the best model.
        """
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'model_state': self._get_model_state(trainer),
            'optimizer_state': self._get_optimizer_state(trainer),
        }
 
        if is_best:
            filepath = self.checkpoint_dir / 'best_checkpoint.pth'
        else:
            filepath = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
 
        try:
            save_pickle(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
 
            if not is_best:
                self.checkpoints.append(filepath)
                self._cleanup_old_checkpoints()
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
 
    def _get_model_state(self, trainer) -> Dict:
        """Get model state dictionary.
 
        Unwraps DDP/DistributedDataParallel to save the underlying model.
        This ensures checkpoints can be loaded in both distributed and
        single-process modes.
        """
        model = trainer.model
 
        # Only unwrap if model is an actual DDP/DataParallel instance
        # This avoids false positives with mock objects in tests
        try:
            from torch.nn.parallel import DistributedDataParallel, DataParallel
            if isinstance(model, (DistributedDataParallel, DataParallel)):
                model = model.module
        except (ImportError, TypeError):
            pass
 
        if hasattr(model, 'state_dict') and callable(model.state_dict):
            return model.state_dict()
        return {}
 
    def _get_optimizer_state(self, trainer) -> Dict:
        """Get optimizer state dictionary."""
        optimizer = trainer.optimizer
        if hasattr(optimizer, 'state_dict'):
            return optimizer.state_dict()
        return {}
 
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints."""
        while len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")
 
    def _strip_module_prefix(self, state_dict: dict) -> dict:
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
 
    def load_checkpoint(self, trainer, checkpoint_path: Union[str, Path]) -> Dict:
        """Load checkpoint.
 
        Args:
            trainer: Trainer instance.
            checkpoint_path: Path to checkpoint.
 
        Returns:
            Checkpoint dictionary.
        """
        from utils.io import load_pickle
 
        checkpoint = load_pickle(checkpoint_path)
 
        # Restore model state
        if 'model_state' in checkpoint and checkpoint['model_state']:
            if hasattr(trainer.model, 'load_state_dict'):
                # Strip 'module.' prefix to handle DDP-wrapped checkpoints
                model_state = self._strip_module_prefix(checkpoint['model_state'])
                trainer.model.load_state_dict(model_state)
 
        # Restore optimizer state
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state']:
            if hasattr(trainer.optimizer, 'load_state_dict'):
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
 
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
 
    def load_best(self, trainer) -> Optional[Dict]:
        """Load best checkpoint.
 
        Args:
            trainer: Trainer instance.
 
        Returns:
            Checkpoint dictionary or None if not found.
        """
        best_path = self.checkpoint_dir / 'best_checkpoint.pth'
        if best_path.exists():
            return self.load_checkpoint(trainer, best_path)
        logger.warning("Best checkpoint not found")
        return None
 
    def load_latest(self, trainer) -> Optional[Dict]:
        """Load latest checkpoint.
 
        Args:
            trainer: Trainer instance.
 
        Returns:
            Checkpoint dictionary or None if not found.
        """
        if not self.checkpoints:
            # Find checkpoints in directory
            checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
                key=lambda p: p.stat().st_mtime
            )
            if checkpoints:
                return self.load_checkpoint(trainer, checkpoints[-1])
        else:
            return self.load_checkpoint(trainer, self.checkpoints[-1])
 
        logger.warning("No checkpoints found")
        return None