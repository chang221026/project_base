"""Model library.
 
Provides model registry and base classes for model implementations.
"""
from utils.registry import Registry
 
# Create model registry
MODELS = Registry('model')
 
# Import base classes
from .base import BaseModel
 
# Import implementations to trigger registration
from . import mlp
 
__all__ = ['MODELS', 'BaseModel']