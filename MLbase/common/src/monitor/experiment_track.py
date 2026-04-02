"""Experiment tracking module.

Provides experiment run management, metric logging, parameter tracking,
artifact management, and environment capture for reproducible ML experiments.
"""
import os
import sys
import time
import uuid
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.logger import get_logger
from utils.io import save_json, load_json, ensure_dir


logger = get_logger()


def _collect_environment_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility.

    Captures Python version, framework versions, hardware info,
    OS details, and git commit hash.

    Returns:
        Dictionary of environment information.
    """
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
    }

    # PyTorch
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    # NumPy
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        pass

    # Git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()

        result_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result_dirty.returncode == 0:
            env["git_dirty"] = len(result_dirty.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return env


class Run:
    """A single experiment run.

    Tracks parameters, metrics (per step and per epoch), artifacts,
    and environment info for one training session.
    """

    def __init__(self,
                 run_id: Optional[str] = None,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        """Initialize a run.

        Args:
            run_id: Unique run identifier. Auto-generated if None.
            name: Human-readable run name.
            tags: Key-value tags for filtering/grouping.
        """
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.name = name or f"run_{self.run_id}"
        self.tags = tags or {}

        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.artifacts: List[Dict[str, str]] = []
        self.environment: Dict[str, Any] = {}

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "created"  # created / running / completed / failed

    def start(self) -> None:
        """Mark run as started and auto-capture environment info."""
        self.start_time = time.time()
        self.status = "running"
        self.environment = _collect_environment_info()
        logger.info(f"Run '{self.name}' [{self.run_id}] started")

    def end(self, status: str = "completed") -> None:
        """Mark run as ended.

        Args:
            status: Final status ('completed' or 'failed').
        """
        self.end_time = time.time()
        self.status = status
        duration = self.end_time - (self.start_time or self.end_time)
        logger.info(
            f"Run '{self.name}' [{self.run_id}] {status} "
            f"in {duration:.2f}s"
        )

    @property
    def duration(self) -> Optional[float]:
        """Run duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value (must be JSON-serializable).
        """
        self.params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            params: Dictionary of parameter name-value pairs.
        """
        self.params.update(params)

    def log_metric(self, key: str, value: float,
                   step: Optional[int] = None,
                   epoch: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Global step number.
            epoch: Epoch number.
        """
        if key not in self.metrics:
            self.metrics[key] = []

        entry = {
            "value": value,
            "timestamp": time.time(),
        }
        if step is not None:
            entry["step"] = step
        if epoch is not None:
            entry["epoch"] = epoch

        self.metrics[key].append(entry)

    def log_metrics(self, metrics: Dict[str, float],
                    step: Optional[int] = None,
                    epoch: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Global step number.
            epoch: Epoch number.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step, epoch=epoch)

    def log_artifact(self, path: str, artifact_type: str = "file",
                     description: str = "") -> None:
        """Log an artifact (file path reference).

        Args:
            path: Path to the artifact file.
            artifact_type: Type of artifact (file, model, plot, data).
            description: Optional description.
        """
        self.artifacts.append({
            "path": str(path),
            "type": artifact_type,
            "description": description,
            "timestamp": time.time(),
        })

    def get_metric(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get all logged values for a metric.

        Args:
            key: Metric name.

        Returns:
            List of metric entries, or None if metric not found.
        """
        return self.metrics.get(key)

    def get_metric_values(self, key: str) -> List[float]:
        """Get just the values for a metric (no metadata).

        Args:
            key: Metric name.

        Returns:
            List of float values.
        """
        entries = self.metrics.get(key, [])
        return [e["value"] for e in entries]

    def get_best_metric(self, key: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """Get the best value for a metric.

        Args:
            key: Metric name.
            mode: 'min' for lowest, 'max' for highest.

        Returns:
            Best metric entry, or None if metric not found.
        """
        entries = self.metrics.get(key)
        if not entries:
            return None
        if mode == "min":
            return min(entries, key=lambda e: e["value"])
        return max(entries, key=lambda e: e["value"])

    def summary(self) -> Dict[str, Any]:
        """Get run summary.

        Returns:
            Dictionary with run metadata, final metrics, and best metrics.
        """
        result = {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "tags": self.tags,
            "duration": self.duration,
            "params": self.params,
            "num_artifacts": len(self.artifacts),
        }

        # Add latest value for each metric
        final_metrics = {}
        for key, entries in self.metrics.items():
            if entries:
                final_metrics[key] = entries[-1]["value"]
        result["final_metrics"] = final_metrics

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize run to dictionary.

        Returns:
            Complete run state as dictionary.
        """
        return {
            "run_id": self.run_id,
            "name": self.name,
            "tags": self.tags,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "environment": self.environment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Run':
        """Deserialize run from dictionary.

        Args:
            data: Run state dictionary.

        Returns:
            Restored Run instance.
        """
        run = cls(run_id=data["run_id"], name=data["name"], tags=data.get("tags"))
        run.params = data.get("params", {})
        run.metrics = data.get("metrics", {})
        run.artifacts = data.get("artifacts", [])
        run.environment = data.get("environment", {})
        run.start_time = data.get("start_time")
        run.end_time = data.get("end_time")
        run.status = data.get("status", "created")
        return run


class ExperimentTracker:
    """Manages multiple experiment runs.

    Provides run lifecycle management, comparison, persistence,
    and automatic environment capture for reproducibility.
    """

    def __init__(self,
                 experiment_name: str = "default",
                 save_dir: Optional[str] = None):
        """Initialize tracker.

        Args:
            experiment_name: Name of the experiment group.
            save_dir: Directory for persisting runs. Defaults to './experiments'.
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir or "./experiments") / experiment_name
        self.runs: Dict[str, Run] = {}
        self.active_run: Optional[Run] = None

        ensure_dir(self.save_dir)
        logger.info(
            f"ExperimentTracker initialized: '{experiment_name}' -> {self.save_dir}"
        )

    def create_run(self,
                   name: Optional[str] = None,
                   tags: Optional[Dict[str, str]] = None) -> Run:
        """Create and start a new run.

        Automatically ends any previous active run. Environment info
        is captured at start time for reproducibility.

        Args:
            name: Run name.
            tags: Run tags.

        Returns:
            The new Run instance (already started).
        """
        if self.active_run and self.active_run.status == "running":
            logger.warning(
                f"Ending previous active run '{self.active_run.name}' before starting new one"
            )
            self.active_run.end()

        run = Run(name=name, tags=tags)
        run.start()
        self.runs[run.run_id] = run
        self.active_run = run
        return run

    def end_run(self, status: str = "completed") -> None:
        """End the active run and persist it.

        Args:
            status: Final status.
        """
        if self.active_run is None:
            logger.warning("No active run to end")
            return
        self.active_run.end(status)
        self._save_run(self.active_run)
        self.active_run = None

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            Run instance or None.
        """
        return self.runs.get(run_id)

    def list_runs(self, status: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> List[Run]:
        """List runs, optionally filtered.

        Args:
            status: Filter by status.
            tags: Filter by tags (all specified tags must match).

        Returns:
            List of matching runs.
        """
        runs = list(self.runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        if tags:
            runs = [r for r in runs
                    if all(r.tags.get(k) == v for k, v in tags.items())]
        return runs

    def compare_runs(self, run_ids: Optional[List[str]] = None,
                     metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: Runs to compare. If None, compares all completed runs.
            metric_keys: Metrics to include. If None, includes all.

        Returns:
            Comparison table as dictionary.
        """
        if run_ids:
            runs = [self.runs[rid] for rid in run_ids if rid in self.runs]
        else:
            runs = [r for r in self.runs.values() if r.status == "completed"]

        if not runs:
            return {"runs": [], "metrics": {}}

        # Collect all metric keys
        all_keys = set()
        for run in runs:
            all_keys.update(run.metrics.keys())

        if metric_keys:
            all_keys = all_keys & set(metric_keys)

        # Build comparison
        comparison = {
            "runs": [],
            "metrics": {key: [] for key in sorted(all_keys)},
        }

        for run in runs:
            comparison["runs"].append({
                "run_id": run.run_id,
                "name": run.name,
                "params": run.params,
                "duration": run.duration,
            })
            for key in sorted(all_keys):
                values = run.get_metric_values(key)
                comparison["metrics"][key].append(
                    values[-1] if values else None
                )

        return comparison

    def get_best_run(self, metric_key: str, mode: str = "min") -> Optional[Run]:
        """Find the run with the best value for a metric.

        Args:
            metric_key: Metric to compare.
            mode: 'min' or 'max'.

        Returns:
            Best run, or None if no runs have the metric.
        """
        best_run = None
        best_value = None

        for run in self.runs.values():
            best = run.get_best_metric(metric_key, mode)
            if best is None:
                continue
            val = best["value"]
            if best_value is None:
                best_value = val
                best_run = run
            elif mode == "min" and val < best_value:
                best_value = val
                best_run = run
            elif mode == "max" and val > best_value:
                best_value = val
                best_run = run

        return best_run

    def _save_run(self, run: Run) -> None:
        """Persist a run to disk.

        Args:
            run: Run to save.
        """
        run_dir = self.save_dir / run.run_id
        ensure_dir(run_dir)
        save_json(run.to_dict(), run_dir / "run.json")
        logger.debug(f"Run '{run.name}' saved to {run_dir}")

    def save_all(self) -> None:
        """Persist all runs to disk."""
        for run in self.runs.values():
            self._save_run(run)
        logger.info(f"Saved {len(self.runs)} runs to {self.save_dir}")

    def load_runs(self) -> int:
        """Load all persisted runs from disk.

        Returns:
            Number of runs loaded.
        """
        count = 0
        if not self.save_dir.exists():
            return count

        for run_dir in self.save_dir.iterdir():
            run_file = run_dir / "run.json"
            if run_file.exists():
                data = load_json(run_file)
                run = Run.from_dict(data)
                self.runs[run.run_id] = run
                count += 1

        logger.info(f"Loaded {count} runs from {self.save_dir}")
        return count

    def delete_run(self, run_id: str) -> bool:
        """Delete a run.

        Args:
            run_id: Run identifier.

        Returns:
            True if run was deleted.
        """
        if run_id not in self.runs:
            return False

        from utils.io import remove_file
        run_dir = self.save_dir / run_id
        if run_dir.exists():
            remove_file(run_dir)

        del self.runs[run_id]
        if self.active_run and self.active_run.run_id == run_id:
            self.active_run = None
        return True

    def export_summary(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export summary of all runs.

        Args:
            output_path: Optional file path to save the summary.

        Returns:
            Summary dictionary.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "total_runs": len(self.runs),
            "runs": [run.summary() for run in self.runs.values()],
        }

        if output_path:
            save_json(summary, output_path)
            logger.info(f"Experiment summary exported to {output_path}")

        return summary
