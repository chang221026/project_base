"""Training hooks module.
 
Provides hooks for training lifecycle management.
"""
from utils.registry import Registry
 
# Create hook registry
HOOKS = Registry('hook')
 
from .base import BaseHook
from .checkpoint import CheckpointHook
from .early_stopping import EarlyStoppingHook
from .lr_scheduler import LRSchedulerHook
from .logging_hook import LoggingHook, TensorBoardHook
 
# Register built-in hooks
HOOKS.register('checkpoint')(CheckpointHook)
HOOKS.register('early_stopping')(EarlyStoppingHook)
HOOKS.register('lr_scheduler')(LRSchedulerHook)
HOOKS.register('logging')(LoggingHook)
HOOKS.register('tensorboard')(TensorBoardHook)
 
__all__ = [
    'HOOKS', 'BaseHook',
    'CheckpointHook', 'EarlyStoppingHook',
    'LRSchedulerHook', 'LoggingHook', 'TensorBoardHook'
]