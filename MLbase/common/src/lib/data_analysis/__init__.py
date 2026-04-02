"""Data analysis library.
 
Provides data analysis and profiling functionality.
"""
from utils.registry import Registry
 
# Create data analysis registry
DATA_ANALYZERS = Registry('data_analyzer')
 
# Import base classes
from .base import BaseDataAnalyzer
 
# Import implementations to trigger registration
from . import simple_analyzer
 
__all__ = ['DATA_ANALYZERS', 'BaseDataAnalyzer']