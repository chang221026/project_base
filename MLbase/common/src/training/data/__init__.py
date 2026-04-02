"""Training data module.
 
Provides data loading and processing utilities.
"""
from utils.registry import Registry
 
# Create data component registries
DATA_PROCESSORS = Registry('data_processor')
DATASET_BUILDERS = Registry('dataset_builder')
DATA_LOADERS = Registry('data_loader')
 
from .dataset_building import Dataset, DatasetBuilder, DataLoader, DataBuilder as _DataBuilder, build_dataloaders
from .data_preprocessing import DataPreprocessingPipeline
 
__all__ = [
    'DATA_PROCESSORS', 'DATASET_BUILDERS', 'DATA_LOADERS',
    'Dataset', 'DatasetBuilder', 'DataLoader',
    'DataBuilder', 'build_dataloaders',
    'DataPreprocessingPipeline'
]
 
# Backward compatibility exports
DataBuilder = _DataBuilder