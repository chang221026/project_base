"""Feature selection library.
 
Provides feature selection and dimensionality reduction functionality.
"""
from utils.registry import Registry
 
# Create feature selection registry
FEATURE_SELECTORS = Registry('feature_selector')
 
# Import base classes
from .base import BaseFeatureSelector
 
# Import implementations to trigger registration
from . import variance_threshold
 
__all__ = ['FEATURE_SELECTORS', 'BaseFeatureSelector']