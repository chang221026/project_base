"""Feature construction library.
 
Provides feature engineering and construction functionality.
"""
from utils.registry import Registry
 
# Create feature construction registry
FEATURE_CONSTRUCTORS = Registry('feature_constructor')
 
# Import base classes
from .base import BaseFeatureConstructor
 
# Import implementations to trigger registration
from . import polynomial
 
__all__ = ['FEATURE_CONSTRUCTORS', 'BaseFeatureConstructor']