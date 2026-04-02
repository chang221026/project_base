"""ML Framework - Enterprise Machine Learning Training and Inference Framework.
 
A unified, efficient, and easy-to-use machine learning development and deployment
solution for teams and production environments.
"""
 
__version__ = '0.1.0'
 
# Utils
from utils.registry import Registry, build_from_cfg
from utils.config_management import Config, load_config
from utils.device_management import DeviceManager, get_device, get_device_manager
from utils.logger import Logger, get_logger, init_logger
from utils.exception import (
    MLFrameworkError,
    ConfigError,
    ModelError,
    DataError,
    TrainingError
)
from utils.io import (
    save_json, load_json,
    save_yaml, load_yaml,
    save_pickle, load_pickle,
    CheckpointManager
)
from utils.distributed_comm import (
    DistributedManager,
    get_dist_manager,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size
)
 
# Libraries
from lib.models import MODELS, BaseModel
from lib.loss_func import LOSSES, BaseLoss
from lib.optimizer import OPTIMIZERS, BaseOptimizer
from lib.evaluator import EVALUATORS, BaseEvaluator
from lib.data_fetching import DATA_FETCHERS, BaseDataFetcher
from lib.data_analysis import DATA_ANALYZERS, BaseDataAnalyzer
from lib.data_processing import DATA_PROCESSORS, BaseDataProcessor
from lib.feature_construction import FEATURE_CONSTRUCTORS, BaseFeatureConstructor
from lib.feature_selection import FEATURE_SELECTORS, BaseFeatureSelector
from lib.imbalance_handling import IMBALANCE_HANDLERS, BaseImbalanceHandler
 
# Training
from training.algorithm.base import BaseAlgorithm
from training.algorithm.supervised import SupervisedAlgorithm
from training.algorithm.unsupervised import UnsupervisedAlgorithm, ClusteringAlgorithm
from training.algorithm.self_supervised import SelfSupervisedAlgorithm, ContrastiveLearning
from training.algorithm.rl import RLAlgorithm, PPO, SAC
from training.data import DATA_LOADERS, DATA_PROCESSORS, DATASET_BUILDERS
from training.data.data_preprocessing import DataPreprocessingPipeline
from training.data.dataset_building import Dataset, DatasetBuilder, DataLoader, DataBuilder, build_dataloaders
from training.hook import HOOKS
from training.hook.base import BaseHook
from training.hook.checkpoint import CheckpointHook
from training.hook.early_stopping import EarlyStoppingHook
from training.hook.lr_scheduler import LRSchedulerHook
from training.hook.logging_hook import LoggingHook
from training.distributed import DistributedEngine, DistributedTrainer

# Monitor
from monitor import (
    ExperimentTracker, Run,
    EvaluationResult, BaseMetric, MetricCollection,
    ConfusionMatrixMetric, ClassificationReport,
    ROCCurveMetric, PRCurveMetric, RegressionMetrics,
    classification_metrics, regression_metrics,
    Profiler, EpochProfiler, Timer, MemoryTracker, ModelAnalyzer,
    TrainingVisualizer,
    ExperimentTrackingHook, ProfilerHook, VisualizationHook,
)

__all__ = [
    # Version
    '__version__',
 
    # Utils
    'Registry', 'build_from_cfg',
    'Config', 'load_config',
    'DeviceManager', 'get_device', 'get_device_manager',
    'Logger', 'get_logger', 'init_logger',
    'MLFrameworkError', 'ConfigError', 'ModelError', 'DataError', 'TrainingError',
    'save_json', 'load_json', 'save_yaml', 'load_yaml', 'save_pickle', 'load_pickle',
    'CheckpointManager',
    'DistributedManager', 'get_dist_manager', 'is_distributed',
    'is_main_process', 'get_rank', 'get_world_size',
 
    # Libraries
    'MODELS', 'BaseModel',
    'LOSSES', 'BaseLoss',
    'OPTIMIZERS', 'BaseOptimizer',
    'EVALUATORS', 'BaseEvaluator',
    'DATA_FETCHERS', 'BaseDataFetcher',
    'DATA_ANALYZERS', 'BaseDataAnalyzer',
    'DATA_PROCESSORS', 'BaseDataProcessor',
    'FEATURE_CONSTRUCTORS', 'BaseFeatureConstructor',
    'FEATURE_SELECTORS', 'BaseFeatureSelector',
    'IMBALANCE_HANDLERS', 'BaseImbalanceHandler',
 
    # Training
    'BaseAlgorithm',
    'SupervisedAlgorithm',
    'UnsupervisedAlgorithm', 'ClusteringAlgorithm',
    'SelfSupervisedAlgorithm', 'ContrastiveLearning',
    'RLAlgorithm', 'DQN', 'PPO', 'SAC',
    'DATA_LOADERS', 'DATA_PROCESSORS', 'DATASET_BUILDERS',
    'DataPreprocessingPipeline',
    'Dataset', 'DatasetBuilder', 'DataLoader', 'DataBuilder', 'build_dataloaders',
    'HOOKS', 'BaseHook', 'CheckpointHook', 'EarlyStoppingHook',
    'LRSchedulerHook', 'LoggingHook',
    'DistributedEngine', 'DistributedTrainer',

    # Monitor
    'ExperimentTracker', 'Run',
    'EvaluationResult', 'BaseMetric', 'MetricCollection',
    'ConfusionMatrixMetric', 'ClassificationReport',
    'ROCCurveMetric', 'PRCurveMetric', 'RegressionMetrics',
    'classification_metrics', 'regression_metrics',
    'Profiler', 'EpochProfiler', 'Timer', 'MemoryTracker', 'ModelAnalyzer',
    'TrainingVisualizer',
    'ExperimentTrackingHook', 'ProfilerHook', 'VisualizationHook',
]