"""Training algorithms module.
 
Provides algorithm registry and base classes for training algorithms.
"""
from utils.registry import Registry
 
# Create algorithm registry
ALGORITHMS = Registry('algorithm')
 
# Import base class
from .base import BaseAlgorithm
 
# Import and register built-in algorithms
from .supervised import SupervisedAlgorithm
from .unsupervised import UnsupervisedAlgorithm
from .self_supervised import SelfSupervisedAlgorithm
from .rl import RLAlgorithm, PPO, SAC
 
# Register built-in algorithms
ALGORITHMS.register('supervised')(SupervisedAlgorithm)
ALGORITHMS.register('unsupervised')(UnsupervisedAlgorithm)
ALGORITHMS.register('self_supervised')(SelfSupervisedAlgorithm)
ALGORITHMS.register('rl')(RLAlgorithm)
ALGORITHMS.register('ppo')(PPO)
ALGORITHMS.register('sac')(SAC)
 
__all__ = [
    'ALGORITHMS',
    'BaseAlgorithm',
    'SupervisedAlgorithm',
    'UnsupervisedAlgorithm',
    'SelfSupervisedAlgorithm',
    'RLAlgorithm',
    'PPO',
    'SAC',
]