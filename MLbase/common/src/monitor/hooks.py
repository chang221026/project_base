"""Monitor hooks module.

Provides training hooks that bridge the monitor layer with the training loop:
- ExperimentTrackingHook: auto-log params/metrics to ExperimentTracker
- ProfilerHook: auto-profile epochs with NaN/gradient health checks
- VisualizationHook: auto-generate plots at training end
"""
import math
from typing import Any, Dict, List, Optional

from training.hook.base import BaseHook
from utils.logger import get_logger


logger = get_logger()


def _is_main_process() -> bool:
    """Check if current process is the main process (rank 0).

    In non-distributed mode, always returns True.
    """
    try:
        from utils.distributed_comm import is_main_process
        return is_main_process()
    except Exception:
        return True


# =============================================================================
# ExperimentTrackingHook
# =============================================================================

class ExperimentTrackingHook(BaseHook):
    """Hook that auto-logs training params, metrics, and artifacts
    to an ExperimentTracker.

    Only logs on rank 0 in distributed training.

    Args:
        experiment_name: Experiment group name.
        save_dir: Directory for experiment data.
        run_name: Optional name for this run.
        run_tags: Optional tags for this run.
        log_params: Whether to auto-log training config as params.
        config: Additional hook configuration.
        priority: Hook priority.
    """

    def __init__(self,
                 experiment_name: str = "default",
                 save_dir: str = "./experiments",
                 run_name: Optional[str] = None,
                 run_tags: Optional[Dict[str, str]] = None,
                 log_params: bool = True,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 0):
        super().__init__(config, priority)
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.run_name = run_name
        self.run_tags = run_tags
        self.log_params = log_params

        self._tracker = None
        self._is_main = True

    def on_train_start(self, trainer) -> None:
        """Create experiment run and log training parameters."""
        self._is_main = _is_main_process()
        if not self._is_main:
            return

        from monitor.experiment_track import ExperimentTracker

        self._tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            save_dir=self.save_dir,
        )
        self._tracker.create_run(name=self.run_name, tags=self.run_tags)

        if self.log_params:
            try:
                training_config = trainer.get_config()
                # Flatten nested config for logging
                flat_params = self._flatten_config(training_config)
                self._tracker.active_run.log_params(flat_params)
            except Exception:
                pass

    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log epoch metrics."""
        if not self._is_main or self._tracker is None:
            return
        if self._tracker.active_run is None:
            return

        self._tracker.active_run.log_metrics(
            metrics,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
        )

    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log validation metrics with 'val_' prefix."""
        if not self._is_main or self._tracker is None:
            return
        if self._tracker.active_run is None:
            return

        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        self._tracker.active_run.log_metrics(
            val_metrics,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
        )

    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """End the run and persist data."""
        if not self._is_main or self._tracker is None:
            return

        self._tracker.end_run(status="completed")

    @property
    def tracker(self):
        """Access the underlying ExperimentTracker."""
        return self._tracker

    @staticmethod
    def _flatten_config(config: Dict[str, Any],
                        prefix: str = "") -> Dict[str, Any]:
        """Flatten nested config dict for param logging.

        Args:
            config: Nested config dictionary.
            prefix: Key prefix for flattening.

        Returns:
            Flat dictionary.
        """
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(
                    ExperimentTrackingHook._flatten_config(value, full_key)
                )
            elif isinstance(value, (list, tuple)):
                flat[full_key] = str(value)
            else:
                flat[full_key] = value
        return flat


# =============================================================================
# ProfilerHook
# =============================================================================

class ProfilerHook(BaseHook):
    """Hook that auto-profiles training epochs and performs health checks.

    Features:
    - Per-epoch timing
    - Optional memory tracking
    - NaN detection in loss
    - Gradient norm monitoring
    - Report generation at training end

    Only profiles on rank 0 in distributed training.

    Args:
        profile_memory: Whether to track memory per epoch.
        detect_nan: Whether to check for NaN in metrics.
        monitor_gradients: Whether to log gradient norms.
        gradient_clip_threshold: Warn if gradient norm exceeds this.
        config: Additional hook configuration.
        priority: Hook priority.
    """

    def __init__(self,
                 profile_memory: bool = False,
                 detect_nan: bool = True,
                 monitor_gradients: bool = False,
                 gradient_clip_threshold: float = 100.0,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 0):
        super().__init__(config, priority)
        self.profile_memory = profile_memory
        self.detect_nan = detect_nan
        self.monitor_gradients = monitor_gradients
        self.gradient_clip_threshold = gradient_clip_threshold

        self._profiler = None
        self._is_main = True
        self._nan_count = 0

    def on_train_start(self, trainer) -> None:
        """Initialize the profiler."""
        self._is_main = _is_main_process()
        if not self._is_main:
            return

        from monitor.performance_analysis import EpochProfiler
        self._profiler = EpochProfiler(enabled=True)

    def on_epoch_start(self, trainer) -> None:
        """Start epoch timing."""
        if not self._is_main or self._profiler is None:
            return
        self._profiler.epoch_start()

        if self.profile_memory:
            self._profiler.profiler.memory_snapshot(
                label=f"epoch_{trainer.current_epoch}_start"
            )

    def on_epoch_end(self, trainer, metrics: Dict[str, float]) -> None:
        """End epoch timing and run health checks."""
        if not self._is_main or self._profiler is None:
            return

        self._profiler.epoch_end()

        if self.profile_memory:
            self._profiler.profiler.memory_snapshot(
                label=f"epoch_{trainer.current_epoch}_end"
            )

        # NaN detection
        if self.detect_nan:
            for key, value in metrics.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    self._nan_count += 1
                    logger.error(
                        f"[HealthCheck] NaN/Inf detected in '{key}' "
                        f"at epoch {trainer.current_epoch + 1} "
                        f"(total NaN occurrences: {self._nan_count})"
                    )

        # Gradient monitoring
        if self.monitor_gradients and trainer.model is not None:
            self._check_gradients(trainer)

    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """Print profiling report."""
        if not self._is_main or self._profiler is None:
            return
        self._profiler.print_report()

    @property
    def profiler(self):
        """Access the underlying EpochProfiler."""
        return self._profiler

    def _check_gradients(self, trainer) -> None:
        """Check gradient norms for health issues."""
        try:
            import torch

            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            if math.isnan(total_norm) or math.isinf(total_norm):
                logger.error(
                    f"[HealthCheck] Gradient NaN/Inf detected "
                    f"at epoch {trainer.current_epoch + 1}"
                )
            elif total_norm > self.gradient_clip_threshold:
                logger.warning(
                    f"[HealthCheck] Large gradient norm: {total_norm:.2f} "
                    f"(threshold: {self.gradient_clip_threshold}) "
                    f"at epoch {trainer.current_epoch + 1}"
                )
        except Exception:
            pass


# =============================================================================
# VisualizationHook
# =============================================================================

class VisualizationHook(BaseHook):
    """Hook that auto-generates training plots at training end.

    Only generates plots on rank 0 in distributed training.

    Args:
        save_dir: Directory for saving plots.
        plot_metrics: List of metric names to plot. If None, plots all.
        style: Matplotlib style name.
        dpi: Plot resolution.
        config: Additional hook configuration.
        priority: Hook priority.
    """

    def __init__(self,
                 save_dir: str = "./plots",
                 plot_metrics: Optional[List[str]] = None,
                 style: str = "default",
                 dpi: int = 150,
                 config: Optional[Dict[str, Any]] = None,
                 priority: int = 100):
        super().__init__(config, priority)
        self.save_dir = save_dir
        self.plot_metrics = plot_metrics
        self.style = style
        self.dpi = dpi
        self._is_main = True

    def on_train_end(self, trainer, history: Dict[str, Any]) -> None:
        """Generate training curve plots."""
        self._is_main = _is_main_process()
        if not self._is_main:
            return

        try:
            from monitor.visualization import TrainingVisualizer

            visualizer = TrainingVisualizer(
                save_dir=self.save_dir,
                style=self.style,
                dpi=self.dpi,
            )

            path = visualizer.plot_training_curves(
                history=history,
                metrics=self.plot_metrics,
                filename="training_curves.png",
            )

            if path:
                logger.info(f"Training curves saved to {path}")

        except ImportError:
            logger.warning(
                "matplotlib not available, skipping visualization. "
                "Install with: pip install matplotlib"
            )
        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")
