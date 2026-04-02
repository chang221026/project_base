"""Training module.
 
Provides unified training entry point and algorithm implementations.
"""
 
# Main training entry point
from .trainer import Trainer, train
 
# Algorithm registry and implementations
from .algorithm import (
    ALGORITHMS,
    BaseAlgorithm,
    SupervisedAlgorithm,
    UnsupervisedAlgorithm,
    SelfSupervisedAlgorithm,
    RLAlgorithm
)
 
# Distributed training
from .distributed.engine import DistributedEngine, DistributedTrainer
 
# Hooks
from .hook import (
    BaseHook,
    CheckpointHook,
    EarlyStoppingHook,
    LRSchedulerHook,
    LoggingHook,
    TensorBoardHook
)
 
# Data utilities
from .data.dataset_building import Dataset, DatasetBuilder
from .data.data_preprocessing import DataPreprocessingPipeline
from .data_facade import DataFacade
 
__all__ = [
    # Main entry point
    'Trainer',
    'train',
 
    # Algorithm
    'ALGORITHMS',
    'BaseAlgorithm',
    'SupervisedAlgorithm',
    'UnsupervisedAlgorithm',
    'SelfSupervisedAlgorithm',
    'RLAlgorithm',
 
    # Distributed
    'DistributedEngine',
    'DistributedTrainer',
 
    # Hooks
    'BaseHook',
    'CheckpointHook',
    'EarlyStoppingHook',
    'LRSchedulerHook',
    'LoggingHook',
    'TensorBoardHook',
 
    # Data
    'Dataset',
    'DatasetBuilder',
    'DataPreprocessingPipeline',
    'DataFacade'
]