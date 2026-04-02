"""Evaluation & Monitoring module.

Provides experiment tracking, post-training evaluation metrics,
performance profiling, visualization, and training hooks for
automatic monitoring integration.

Module structure:
    - metrics: Post-training comprehensive evaluation (confusion matrix, ROC, PR, regression)
    - experiment_track: Experiment run management, persistence, comparison
    - performance_analysis: Runtime profiling, model complexity analysis, inference benchmarking
    - visualization: Training curves, evaluation plots, profiling charts
    - hooks: Training hooks for automatic monitoring (ExperimentTracking, Profiler, Visualization)
"""

# Experiment Tracking
from .experiment_track import ExperimentTracker, Run

# Metrics
from .metrics import (
    EvaluationResult, BaseMetric, MetricCollection,
    ConfusionMatrixMetric, ClassificationReport,
    ROCCurveMetric, PRCurveMetric, RegressionMetrics,
    classification_metrics, regression_metrics,
)

# Performance Analysis
from .performance_analysis import (
    Profiler, EpochProfiler, Timer, MemoryTracker, ModelAnalyzer,
)

# Visualization
from .visualization import TrainingVisualizer

# Hooks
from .hooks import (
    ExperimentTrackingHook, ProfilerHook, VisualizationHook,
)

__all__ = [
    # Experiment Tracking
    'ExperimentTracker', 'Run',

    # Metrics
    'EvaluationResult', 'BaseMetric', 'MetricCollection',
    'ConfusionMatrixMetric', 'ClassificationReport',
    'ROCCurveMetric', 'PRCurveMetric', 'RegressionMetrics',
    'classification_metrics', 'regression_metrics',

    # Performance Analysis
    'Profiler', 'EpochProfiler', 'Timer', 'MemoryTracker', 'ModelAnalyzer',

    # Visualization
    'TrainingVisualizer',

    # Hooks
    'ExperimentTrackingHook', 'ProfilerHook', 'VisualizationHook',
]
