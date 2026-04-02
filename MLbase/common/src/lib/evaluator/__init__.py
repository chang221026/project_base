"""Evaluator library.
 
Provides evaluator registry and implementations for model evaluation.
"""
from utils.registry import Registry
 
# Create evaluator registry
EVALUATORS = Registry('evaluator')
 
# Import base classes
from .base import BaseEvaluator
 
# Import implementations to trigger registration
from . import accuracy
 
__all__ = ['EVALUATORS', 'BaseEvaluator']