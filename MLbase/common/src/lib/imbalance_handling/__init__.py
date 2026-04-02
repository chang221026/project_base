"""Imbalance handling library.
 
Provides methods for handling imbalanced datasets.
"""
from utils.registry import Registry
 
# Create imbalance handling registry
IMBALANCE_HANDLERS = Registry('imbalance_handler')
 
# Import base classes
from .base import BaseImbalanceHandler
 
# Import implementations to trigger registration
from . import random_oversampler
 
__all__ = ['IMBALANCE_HANDLERS', 'BaseImbalanceHandler']