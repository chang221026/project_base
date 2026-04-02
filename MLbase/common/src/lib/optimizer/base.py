"""Base optimizer class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
 
 
class BaseOptimizer(ABC):
    """Base class for all optimizers.
 
    All optimizers should inherit from this class and implement
    the required abstract methods.
    """
 
    def __init__(self, parameters, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize optimizer.
 
        Args:
            parameters: Model parameters to optimize.
            config: Optimizer configuration dictionary.
            **kwargs: Additional configuration parameters.
        """
        self.parameters = list(parameters) if parameters else []
        self.config = config or {}
        self.config.update(kwargs)
        self.state = {}
        self._step_count = 0
 
    @abstractmethod
    def step(self) -> None:
        """Perform one optimization step."""
        pass
 
    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out gradients."""
        pass
 
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state.
 
        Returns:
            State dictionary.
        """
        return {
            'state': self.state,
            'config': self.config,
            'step_count': self._step_count
        }
 
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state.
 
        Args:
            state_dict: State dictionary to load.
        """
        self.state = state_dict.get('state', {})
        self.config.update(state_dict.get('config', {}))
        self._step_count = state_dict.get('step_count', 0)
 
    def get_lr(self) -> float:
        """Get current learning rate.
 
        Returns:
            Learning rate.
        """
        return self.config.get('lr', 0.001)
 
    def set_lr(self, lr: float) -> None:
        """Set learning rate.
 
        Args:
            lr: New learning rate.
        """
        self.config['lr'] = lr
 
    @property
    def step_count(self) -> int:
        """Get step count."""
        return self._step_count