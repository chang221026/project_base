"""Performance analysis module.

Provides runtime profiling (time, memory, throughput), model complexity
analysis (parameter count, model size, FLOPs), and inference benchmarking.
"""
import time
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger


logger = get_logger()


# =============================================================================
# Timer
# =============================================================================

class Timer:
    """High-precision timer for measuring code execution time.

    Supports both context manager and manual start/stop usage.
    """

    def __init__(self):
        self._start: Optional[float] = None
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> 'Timer':
        """Start the timer."""
        if not self._running:
            self._start = time.perf_counter()
            self._running = True
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        if self._running and self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._running = False
            self._start = None
        return self._elapsed

    def reset(self) -> None:
        """Reset the timer."""
        self._start = None
        self._elapsed = 0.0
        self._running = False

    @property
    def elapsed(self) -> float:
        """Current elapsed time in seconds."""
        if self._running and self._start is not None:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    def __enter__(self) -> 'Timer':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# =============================================================================
# MemoryTracker
# =============================================================================

class MemoryTracker:
    """Tracks memory usage during training.

    Supports both system memory (via psutil) and GPU memory (via torch.cuda).
    Gracefully degrades when optional dependencies are unavailable.
    """

    def __init__(self):
        self._snapshots: List[Dict[str, Any]] = []
        self._has_psutil = False
        self._has_cuda = False

        try:
            import psutil
            self._has_psutil = True
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                self._has_cuda = True
        except ImportError:
            pass

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot.

        Args:
            label: Optional label for this snapshot.

        Returns:
            Memory usage dictionary.
        """
        info: Dict[str, Any] = {
            "label": label,
            "timestamp": time.time(),
        }

        if self._has_psutil:
            import psutil
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            info["system"] = {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
            }

        if self._has_cuda:
            import torch
            info["gpu"] = {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }

        self._snapshots.append(info)
        return info

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage across all snapshots.

        Returns:
            Dictionary with peak memory values in MB.
        """
        result: Dict[str, float] = {}

        system_peaks = [
            s["system"]["rss_mb"] for s in self._snapshots if "system" in s
        ]
        if system_peaks:
            result["system_peak_rss_mb"] = max(system_peaks)

        gpu_peaks = [
            s["gpu"]["max_allocated_mb"] for s in self._snapshots if "gpu" in s
        ]
        if gpu_peaks:
            result["gpu_peak_allocated_mb"] = max(gpu_peaks)

        return result

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all memory snapshots."""
        return self._snapshots.copy()

    def reset(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()


# =============================================================================
# Profiler
# =============================================================================

class Profiler:
    """Training profiler for performance analysis.

    Tracks timing, memory, and throughput across training phases
    (epoch, batch, forward, backward, data loading, etc.).
    """

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is active. When disabled,
                     all operations become no-ops for zero overhead.
        """
        self.enabled = enabled
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._active_timers: Dict[str, Timer] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._memory = MemoryTracker()
        self._throughput_data: List[Dict[str, Any]] = []
        self._total_samples: int = 0
        self._total_time: float = 0.0

    def start(self, name: str) -> None:
        """Start timing a named section.

        Args:
            name: Section name (e.g., 'epoch', 'forward', 'backward').
        """
        if not self.enabled:
            return
        timer = Timer()
        timer.start()
        self._active_timers[name] = timer

    def stop(self, name: str) -> float:
        """Stop timing a named section.

        Args:
            name: Section name.

        Returns:
            Elapsed time for this section in seconds.
        """
        if not self.enabled:
            return 0.0
        timer = self._active_timers.pop(name, None)
        if timer is None:
            return 0.0
        elapsed = timer.stop()
        self._timers[name].append(elapsed)
        self._counters[name] += 1
        return elapsed

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code section.

        Args:
            name: Section name.

        Example:
            with profiler.profile('forward'):
                output = model(inputs)
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def record_throughput(self, num_samples: int, elapsed: float,
                          epoch: Optional[int] = None) -> None:
        """Record throughput data.

        Args:
            num_samples: Number of samples processed.
            elapsed: Time taken in seconds.
            epoch: Optional epoch number.
        """
        if not self.enabled or elapsed <= 0:
            return

        self._total_samples += num_samples
        self._total_time += elapsed

        self._throughput_data.append({
            "num_samples": num_samples,
            "elapsed": elapsed,
            "samples_per_sec": num_samples / elapsed,
            "epoch": epoch,
            "timestamp": time.time(),
        })

    def memory_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot.

        Args:
            label: Snapshot label.

        Returns:
            Memory usage info.
        """
        if not self.enabled:
            return {}
        return self._memory.snapshot(label)

    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics for a named section.

        Args:
            name: Section name.

        Returns:
            Dictionary with count, total, mean, min, max times.
        """
        times = self._timers.get(name, [])
        if not times:
            return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    def get_all_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all sections.

        Returns:
            Dictionary mapping section names to their stats.
        """
        return {name: self.get_timing_stats(name) for name in self._timers}

    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics.

        Returns:
            Dictionary with throughput metrics.
        """
        if not self._throughput_data:
            return {"total_samples": 0, "total_time": 0.0, "avg_samples_per_sec": 0.0}

        rates = [d["samples_per_sec"] for d in self._throughput_data]
        return {
            "total_samples": self._total_samples,
            "total_time": self._total_time,
            "avg_samples_per_sec": (
                self._total_samples / self._total_time if self._total_time > 0 else 0.0
            ),
            "min_samples_per_sec": min(rates),
            "max_samples_per_sec": max(rates),
        }

    def report(self) -> Dict[str, Any]:
        """Generate a full profiling report.

        Returns:
            Complete profiling report dictionary.
        """
        report = {
            "timing": self.get_all_timing_stats(),
            "throughput": self.get_throughput_stats(),
            "memory": {
                "peak": self._memory.get_peak_memory(),
                "num_snapshots": len(self._memory.get_snapshots()),
            },
        }

        # Identify bottlenecks: sections ranked by total time
        if report["timing"]:
            sorted_sections = sorted(
                report["timing"].items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )
            total_profiled_time = sum(s[1]["total"] for s in sorted_sections)
            if total_profiled_time > 0:
                report["bottlenecks"] = [
                    {
                        "section": name,
                        "total_time": stats["total"],
                        "percentage": (stats["total"] / total_profiled_time) * 100,
                        "call_count": stats["count"],
                        "avg_time": stats["mean"],
                    }
                    for name, stats in sorted_sections[:5]
                ]

        return report

    def print_report(self) -> None:
        """Print a formatted profiling report to the logger."""
        report = self.report()

        logger.info("=" * 60)
        logger.info("Performance Profiling Report")
        logger.info("=" * 60)

        if report["timing"]:
            logger.info("\n--- Timing ---")
            for name, stats in sorted(
                report["timing"].items(),
                key=lambda x: x[1]["total"],
                reverse=True
            ):
                logger.info(
                    f"  {name:<25s} | total={stats['total']:.4f}s | "
                    f"count={stats['count']} | mean={stats['mean']:.4f}s | "
                    f"min={stats['min']:.4f}s | max={stats['max']:.4f}s"
                )

        tp = report["throughput"]
        if tp["total_samples"] > 0:
            logger.info("\n--- Throughput ---")
            logger.info(f"  Total samples: {tp['total_samples']}")
            logger.info(f"  Total time: {tp['total_time']:.2f}s")
            logger.info(f"  Avg throughput: {tp['avg_samples_per_sec']:.1f} samples/s")

        mem = report["memory"]
        if mem["peak"]:
            logger.info("\n--- Memory ---")
            for key, val in mem["peak"].items():
                logger.info(f"  {key}: {val:.1f} MB")

        if "bottlenecks" in report:
            logger.info("\n--- Bottlenecks ---")
            for b in report["bottlenecks"]:
                logger.info(
                    f"  {b['section']:<25s} | "
                    f"{b['percentage']:.1f}% of profiled time | "
                    f"{b['total_time']:.4f}s total"
                )

        logger.info("=" * 60)

    def reset(self) -> None:
        """Reset all profiling data."""
        self._timers.clear()
        self._active_timers.clear()
        self._counters.clear()
        self._memory.reset()
        self._throughput_data.clear()
        self._total_samples = 0
        self._total_time = 0.0


# =============================================================================
# EpochProfiler
# =============================================================================

class EpochProfiler:
    """Convenience profiler that tracks per-epoch timing breakdown.

    Automatically profiles phases within each epoch (forward, backward,
    data loading, etc.) and records throughput.
    """

    def __init__(self, enabled: bool = True):
        """Initialize epoch profiler.

        Args:
            enabled: Whether profiling is active.
        """
        self.profiler = Profiler(enabled=enabled)
        self._epoch_times: List[Dict[str, float]] = []
        self._current_epoch_timings: Dict[str, float] = {}

    def epoch_start(self) -> None:
        """Mark epoch start."""
        self.profiler.start("epoch")
        self._current_epoch_timings = {}

    def epoch_end(self, num_samples: int = 0) -> Dict[str, float]:
        """Mark epoch end and record throughput.

        Args:
            num_samples: Number of samples processed in this epoch.

        Returns:
            Timing breakdown for this epoch.
        """
        epoch_time = self.profiler.stop("epoch")
        self._current_epoch_timings["epoch_total"] = epoch_time

        if num_samples > 0:
            self.profiler.record_throughput(
                num_samples, epoch_time,
                epoch=len(self._epoch_times)
            )

        self._epoch_times.append(self._current_epoch_timings.copy())
        return self._current_epoch_timings

    def phase(self, name: str):
        """Context manager for profiling a phase within an epoch.

        Args:
            name: Phase name (e.g., 'forward', 'backward', 'data_load').

        Returns:
            Context manager.
        """
        return self.profiler.profile(name)

    def step_start(self, name: str) -> None:
        """Start timing a named step."""
        self.profiler.start(name)

    def step_end(self, name: str) -> float:
        """End timing a named step."""
        elapsed = self.profiler.stop(name)
        self._current_epoch_timings[name] = (
            self._current_epoch_timings.get(name, 0.0) + elapsed
        )
        return elapsed

    def get_epoch_times(self) -> List[Dict[str, float]]:
        """Get timing breakdown for all epochs."""
        return self._epoch_times.copy()

    def report(self) -> Dict[str, Any]:
        """Generate profiling report."""
        report = self.profiler.report()
        report["epoch_breakdown"] = self._epoch_times
        return report

    def print_report(self) -> None:
        """Print profiling report."""
        self.profiler.print_report()

        if self._epoch_times:
            logger.info("\n--- Per-Epoch Breakdown ---")
            for i, timings in enumerate(self._epoch_times):
                parts = " | ".join(
                    f"{k}={v:.4f}s" for k, v in timings.items()
                )
                logger.info(f"  Epoch {i + 1}: {parts}")

    def reset(self) -> None:
        """Reset all data."""
        self.profiler.reset()
        self._epoch_times.clear()
        self._current_epoch_timings.clear()


# =============================================================================
# ModelAnalyzer
# =============================================================================

class ModelAnalyzer:
    """Static analysis of model complexity and inference performance.

    Provides parameter counting, model size estimation, FLOPs estimation,
    and inference latency benchmarking.
    """

    @staticmethod
    def count_parameters(model) -> Dict[str, int]:
        """Count model parameters.

        Args:
            model: PyTorch model (nn.Module).

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    @staticmethod
    def estimate_model_size(model) -> Dict[str, float]:
        """Estimate model size in memory.

        Args:
            model: PyTorch model (nn.Module).

        Returns:
            Dictionary with size estimates in MB.
        """
        param_size = 0
        buffer_size = 0

        for p in model.parameters():
            param_size += p.nelement() * p.element_size()
        for b in model.buffers():
            buffer_size += b.nelement() * b.element_size()

        total_bytes = param_size + buffer_size

        return {
            "params_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024),
        }

    @staticmethod
    def estimate_flops(model, input_shape: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        """Estimate FLOPs for a forward pass.

        Requires thop or fvcore. Returns None if neither is available.

        Args:
            model: PyTorch model (nn.Module).
            input_shape: Input tensor shape (including batch dim).

        Returns:
            Dictionary with FLOPs info, or None if estimation unavailable.
        """
        import torch

        dummy_input = torch.randn(*input_shape)

        # Move to same device as model
        try:
            device = next(model.parameters()).device
            dummy_input = dummy_input.to(device)
        except StopIteration:
            pass

        # Try thop first
        try:
            from thop import profile as thop_profile
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
            return {
                "flops": int(flops),
                "flops_readable": f"{flops / 1e9:.2f} GFLOPs" if flops >= 1e9
                    else f"{flops / 1e6:.2f} MFLOPs",
                "source": "thop",
            }
        except ImportError:
            pass

        # Try fvcore
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, dummy_input)
            flops = fca.total()
            return {
                "flops": int(flops),
                "flops_readable": f"{flops / 1e9:.2f} GFLOPs" if flops >= 1e9
                    else f"{flops / 1e6:.2f} MFLOPs",
                "source": "fvcore",
            }
        except ImportError:
            pass

        logger.warning(
            "FLOPs estimation unavailable. Install thop or fvcore: "
            "pip install thop  or  pip install fvcore"
        )
        return None

    @staticmethod
    def benchmark_inference(model, input_shape: Tuple[int, ...],
                            num_runs: int = 100,
                            warmup_runs: int = 10,
                            device: Optional[str] = None) -> Dict[str, float]:
        """Benchmark model inference latency.

        Runs the model multiple times and reports latency statistics.

        Args:
            model: PyTorch model (nn.Module).
            input_shape: Input tensor shape (including batch dim).
            num_runs: Number of timed runs.
            warmup_runs: Number of warmup runs (not timed).
            device: Device string. If None, uses model's current device.

        Returns:
            Dictionary with latency stats in milliseconds.
        """
        import torch

        # Determine device
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except StopIteration:
                device = "cpu"

        dummy_input = torch.randn(*input_shape, device=device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Synchronize if on GPU
        if "cuda" in device:
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if "cuda" in device:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy_input)
                if "cuda" in device:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # ms

        latencies.sort()
        n = len(latencies)

        return {
            "num_runs": num_runs,
            "mean_ms": sum(latencies) / n,
            "median_ms": latencies[n // 2],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "p95_ms": latencies[int(n * 0.95)],
            "p99_ms": latencies[int(n * 0.99)],
            "throughput_fps": 1000.0 / (sum(latencies) / n) * input_shape[0],
        }

    @staticmethod
    def summary(model, input_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """Generate a comprehensive model analysis summary.

        Args:
            model: PyTorch model (nn.Module).
            input_shape: Optional input shape for FLOPs estimation.

        Returns:
            Complete model analysis dictionary.
        """
        result = {
            "parameters": ModelAnalyzer.count_parameters(model),
            "size": ModelAnalyzer.estimate_model_size(model),
        }

        if input_shape:
            flops = ModelAnalyzer.estimate_flops(model, input_shape)
            if flops:
                result["flops"] = flops

        return result
