"""Visualization module.

Provides training curve plotting, evaluation result visualization
(confusion matrix, ROC, PR curves), profiling charts, and
experiment comparison plots.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.logger import get_logger
from utils.io import ensure_dir


logger = get_logger()


def _get_matplotlib():
    """Get matplotlib with non-interactive backend fallback.

    Returns:
        (plt, matplotlib) tuple.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib
    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, matplotlib


class TrainingVisualizer:
    """Generates training and evaluation visualization plots.

    Supports training curves, evaluation charts (confusion matrix, ROC, PR),
    profiling visualizations, and multi-run comparisons. Plots can be saved
    to files or displayed interactively.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 style: str = "default",
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 150):
        """Initialize visualizer.

        Args:
            save_dir: Directory for saving plots. Defaults to './plots'.
            style: Matplotlib style name.
            figsize: Default figure size (width, height).
            dpi: Figure resolution.
        """
        self.save_dir = Path(save_dir or "./plots")
        self.style = style
        self.figsize = figsize
        self.dpi = dpi

        ensure_dir(self.save_dir)

    def _create_figure(self, figsize: Optional[Tuple[int, int]] = None):
        """Create a new figure with the configured style.

        Args:
            figsize: Override figure size.

        Returns:
            (fig, ax) tuple.
        """
        plt, _ = _get_matplotlib()
        if self.style != "default":
            try:
                plt.style.use(self.style)
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        return fig, ax

    def _save_or_show(self, fig, filename: Optional[str] = None,
                      show: bool = False) -> Optional[str]:
        """Save figure to file and/or display it.

        Args:
            fig: Matplotlib figure.
            filename: Output filename (saved under save_dir).
            show: Whether to call plt.show().

        Returns:
            Path to saved file, or None.
        """
        plt, _ = _get_matplotlib()
        saved_path = None

        if filename:
            path = self.save_dir / filename
            fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight")
            saved_path = str(path)
            logger.info(f"Plot saved: {saved_path}")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path

    # =========================================================================
    # Training Curves
    # =========================================================================

    def plot_training_curves(self,
                             history: Dict[str, List[Dict[str, float]]],
                             metrics: Optional[List[str]] = None,
                             filename: Optional[str] = "training_curves.png",
                             show: bool = False,
                             title: str = "Training Curves") -> Optional[str]:
        """Plot training and validation curves from training history.

        Compatible with BaseAlgorithm.fit() return format:
        history = {'train': [{'loss': 0.5, ...}, ...], 'val': [...]}

        Args:
            history: Training history dict.
            metrics: Specific metrics to plot. If None, plots all.
            filename: Output filename.
            show: Whether to display the plot.
            title: Plot title.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()

        train_history = history.get("train", [])
        val_history = history.get("val", [])

        if not train_history:
            logger.warning("No training history to plot")
            return None

        # Determine metrics to plot
        all_keys = set()
        for entry in train_history:
            all_keys.update(entry.keys())
        if metrics:
            plot_keys = [k for k in metrics if k in all_keys]
        else:
            plot_keys = sorted(all_keys)

        if not plot_keys:
            logger.warning("No metrics found to plot")
            return None

        n_metrics = len(plot_keys)
        n_cols = min(n_metrics, 2)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2),
            squeeze=False
        )

        epochs = list(range(1, len(train_history) + 1))

        for idx, key in enumerate(plot_keys):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row][col]

            train_vals = [e.get(key, 0.0) for e in train_history]
            ax.plot(epochs, train_vals, "b-", label=f"Train {key}", linewidth=1.5)

            if val_history:
                val_vals = [e.get(key, 0.0) for e in val_history]
                ax.plot(
                    epochs[:len(val_vals)], val_vals,
                    "r--", label=f"Val {key}", linewidth=1.5
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row][col].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        return self._save_or_show(fig, filename, show)

    def plot_metric(self,
                    values: List[float],
                    name: str = "metric",
                    steps: Optional[List[int]] = None,
                    filename: Optional[str] = None,
                    show: bool = False,
                    xlabel: str = "Step",
                    ylabel: Optional[str] = None,
                    title: Optional[str] = None,
                    markers: Optional[Dict[str, int]] = None) -> Optional[str]:
        """Plot a single metric over time.

        Args:
            values: Metric values.
            name: Metric name.
            steps: X-axis values. Defaults to 1..N.
            filename: Output filename.
            show: Whether to display.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            markers: Dict of {label: step_index} to mark special points.

        Returns:
            Path to saved file.
        """
        fig, ax = self._create_figure()

        x = steps or list(range(1, len(values) + 1))
        ax.plot(x, values, "b-", linewidth=1.5, label=name)

        if markers:
            for label, step_idx in markers.items():
                if 0 <= step_idx < len(values):
                    ax.axvline(x=x[step_idx], color="gray", linestyle=":", alpha=0.5)
                    ax.annotate(
                        label,
                        xy=(x[step_idx], values[step_idx]),
                        xytext=(5, 10), textcoords="offset points",
                        fontsize=8, color="red"
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or name)
        ax.set_title(title or f"{name} over {xlabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, filename or f"{name}.png", show)

    def plot_comparison(self,
                        data: Dict[str, List[float]],
                        metric_name: str = "metric",
                        filename: Optional[str] = None,
                        show: bool = False,
                        title: Optional[str] = None) -> Optional[str]:
        """Plot multiple runs/series on the same chart for comparison.

        Args:
            data: Dictionary mapping series name -> values.
            metric_name: Name of the metric being compared.
            filename: Output filename.
            show: Whether to display.
            title: Plot title.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()
        fig, ax = self._create_figure()

        colors = plt.cm.tab10.colors
        for i, (name, values) in enumerate(data.items()):
            color = colors[i % len(colors)]
            x = list(range(1, len(values) + 1))
            ax.plot(x, values, color=color, linewidth=1.5, label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(title or f"{metric_name} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_or_show(
            fig, filename or f"{metric_name}_comparison.png", show
        )

    def plot_learning_rate(self,
                           lr_values: List[float],
                           filename: Optional[str] = "learning_rate.png",
                           show: bool = False) -> Optional[str]:
        """Plot learning rate schedule.

        Args:
            lr_values: Learning rate values per epoch/step.
            filename: Output filename.
            show: Whether to display.

        Returns:
            Path to saved file.
        """
        return self.plot_metric(
            values=lr_values,
            name="Learning Rate",
            xlabel="Epoch",
            ylabel="Learning Rate",
            title="Learning Rate Schedule",
            filename=filename,
            show=show,
        )

    # =========================================================================
    # Evaluation Plots
    # =========================================================================

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              normalize: bool = False,
                              filename: Optional[str] = "confusion_matrix.png",
                              show: bool = False,
                              title: str = "Confusion Matrix",
                              cmap: str = "Blues") -> Optional[str]:
        """Plot confusion matrix as a heatmap.

        Args:
            cm: Confusion matrix (N x N numpy array).
            class_names: Class label names.
            normalize: Whether to normalize by row (true label).
            filename: Output filename.
            show: Whether to display.
            title: Plot title.
            cmap: Colormap name.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
            fmt = ".2f"
        else:
            cm_display = cm
            fmt = "d"

        n_classes = cm.shape[0]
        labels = class_names or [str(i) for i in range(n_classes)]

        fig, ax = self._create_figure(
            figsize=(max(6, n_classes * 0.8), max(5, n_classes * 0.7))
        )

        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Annotate cells
        thresh = cm_display.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_display[i, j]
                text = f"{val:{fmt}}" if fmt == "d" else f"{val:.2f}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8 if n_classes > 10 else 10
                )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        fig.tight_layout()

        return self._save_or_show(fig, filename, show)

    def plot_roc_curve(self,
                       roc_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                       filename: Optional[str] = "roc_curve.png",
                       show: bool = False,
                       title: str = "ROC Curve") -> Optional[str]:
        """Plot ROC curve(s).

        Args:
            roc_data: Single ROC dict {'fpr': [...], 'tpr': [...]} or
                      list of {'name': ..., 'fpr': ..., 'tpr': ...}.
            filename: Output filename.
            show: Whether to display.
            title: Plot title.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()
        fig, ax = self._create_figure()

        if isinstance(roc_data, dict):
            roc_data = [roc_data]

        colors = plt.cm.tab10.colors
        for i, curve in enumerate(roc_data):
            fpr = curve["fpr"]
            tpr = curve["tpr"]
            label = curve.get("name", f"Class {i}")

            # Compute AUC if not provided
            auc_val = curve.get("auc")
            if auc_val is None:
                auc_val = float(np.trapz(
                    np.array(tpr)[np.argsort(fpr)],
                    np.array(fpr)[np.argsort(fpr)]
                ))

            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linewidth=1.5,
                label=f"{label} (AUC={auc_val:.3f})"
            )

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, filename, show)

    def plot_pr_curve(self,
                      pr_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                      filename: Optional[str] = "pr_curve.png",
                      show: bool = False,
                      title: str = "Precision-Recall Curve") -> Optional[str]:
        """Plot Precision-Recall curve(s).

        Args:
            pr_data: Single PR dict {'precision': [...], 'recall': [...]} or
                     list of {'name': ..., 'precision': ..., 'recall': ...}.
            filename: Output filename.
            show: Whether to display.
            title: Plot title.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()
        fig, ax = self._create_figure()

        if isinstance(pr_data, dict):
            pr_data = [pr_data]

        colors = plt.cm.tab10.colors
        for i, curve in enumerate(pr_data):
            precision = curve["precision"]
            recall = curve["recall"]
            label = curve.get("name", f"Class {i}")

            ap = curve.get("ap")
            if ap is None:
                ap = float(np.trapz(
                    np.array(precision)[np.argsort(recall)],
                    np.array(recall)[np.argsort(recall)]
                ))

            ax.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linewidth=1.5,
                label=f"{label} (AP={ap:.3f})"
            )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, filename, show)

    # =========================================================================
    # Profiling & Memory Plots
    # =========================================================================

    def plot_profiling_report(self,
                              report: Dict[str, Any],
                              filename: Optional[str] = "profiling.png",
                              show: bool = False) -> Optional[str]:
        """Visualize profiling report from Profiler.report().

        Args:
            report: Output of Profiler.report().
            filename: Output filename.
            show: Whether to display.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()

        timing = report.get("timing", {})
        if not timing:
            logger.warning("No timing data to visualize")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart of time distribution
        names = []
        totals = []
        for name, stats in sorted(
            timing.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            names.append(name)
            totals.append(stats["total"])

        if totals:
            ax1.pie(totals, labels=names, autopct="%1.1f%%", startangle=90)
            ax1.set_title("Time Distribution")

        # Bar chart of average times
        means = [timing[n]["mean"] for n in names]
        y_pos = range(len(names))
        ax2.barh(list(y_pos), means, color="steelblue", alpha=0.8)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(names)
        ax2.set_xlabel("Average Time (s)")
        ax2.set_title("Average Time per Section")

        fig.suptitle("Performance Profile", fontsize=14, fontweight="bold")
        fig.tight_layout()

        return self._save_or_show(fig, filename, show)

    def plot_memory_usage(self,
                          snapshots: List[Dict[str, Any]],
                          filename: Optional[str] = "memory_usage.png",
                          show: bool = False) -> Optional[str]:
        """Plot memory usage over time from MemoryTracker snapshots.

        Args:
            snapshots: List of memory snapshots.
            filename: Output filename.
            show: Whether to display.

        Returns:
            Path to saved file.
        """
        if not snapshots:
            logger.warning("No memory snapshots to plot")
            return None

        plt, _ = _get_matplotlib()
        fig, ax = self._create_figure()

        indices = list(range(len(snapshots)))

        sys_mem = [s["system"]["rss_mb"] for s in snapshots if "system" in s]
        if sys_mem:
            ax.plot(indices[:len(sys_mem)], sys_mem, "b-o",
                    label="System RSS (MB)", markersize=3)

        gpu_mem = [s["gpu"]["allocated_mb"] for s in snapshots if "gpu" in s]
        if gpu_mem:
            ax.plot(indices[:len(gpu_mem)], gpu_mem, "r-s",
                    label="GPU Allocated (MB)", markersize=3)

        ax.set_xlabel("Snapshot")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, filename, show)

    # =========================================================================
    # Experiment Comparison
    # =========================================================================

    def plot_experiment_comparison(self,
                                  comparison: Dict[str, Any],
                                  filename: Optional[str] = "experiment_comparison.png",
                                  show: bool = False) -> Optional[str]:
        """Plot comparison from ExperimentTracker.compare_runs() output.

        Args:
            comparison: Output of ExperimentTracker.compare_runs().
            filename: Output filename.
            show: Whether to display.

        Returns:
            Path to saved file.
        """
        plt, _ = _get_matplotlib()

        runs = comparison.get("runs", [])
        metrics = comparison.get("metrics", {})

        if not runs or not metrics:
            logger.warning("No data to compare")
            return None

        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(5 * n_metrics, 6),
            squeeze=False
        )

        run_names = [r["name"] for r in runs]
        x = range(len(run_names))

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[0][idx]
            bar_values = [v if v is not None else 0 for v in values]
            bars = ax.bar(x, bar_values, color="steelblue", alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)

            for bar, val in zip(bars, bar_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7
                )

        fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()

        return self._save_or_show(fig, filename, show)
