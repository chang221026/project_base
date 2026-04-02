"""Loss function library.
 
Provides loss function registry and implementations.
"""
from utils.registry import Registry
 
# Create loss registry
LOSSES = Registry('loss')
 
# Import base classes
from .base import BaseLoss
 
# Import implementations to trigger registration
from . import cross_entropy
 
__all__ = ['LOSSES', 'BaseLoss']