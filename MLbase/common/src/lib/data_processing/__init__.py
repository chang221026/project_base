"""Data processing library.
 
Provides data preprocessing and transformation functionality.
"""
from utils.registry import Registry
 
# Create data processing registry
DATA_PROCESSORS = Registry('data_processor')
 
# Import base classes
from .base import BaseDataProcessor
 
# Import implementations to trigger registration
from . import standard_scaler
 
__all__ = ['DATA_PROCESSORS', 'BaseDataProcessor']