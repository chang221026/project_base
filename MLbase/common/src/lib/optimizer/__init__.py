"""Optimizer library.
 
Provides optimizer registry and implementations.
"""
from utils.registry import Registry
 
# Create optimizer registry
OPTIMIZERS = Registry('optimizer')
 
# Import base classes
from .base import BaseOptimizer
 
# Import implementations to trigger registration
from . import adam
 
__all__ = ['OPTIMIZERS', 'BaseOptimizer']