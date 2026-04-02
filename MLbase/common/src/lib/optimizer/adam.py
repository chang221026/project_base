"""Adam optimizer implementation."""
from typing import Any, Dict, List, Optional, Union
 
import torch
from torch.optim import Adam as TorchAdam
 
from lib.optimizer import OPTIMIZERS, BaseOptimizer
 
 
@OPTIMIZERS.register('Adam')
class Adam(BaseOptimizer):
    """Adam optimizer wrapper.
 
    Wraps PyTorch's Adam optimizer with configuration support.
 
    Config parameters:
        lr: Learning rate (default: 0.001).
        betas: Coefficients for computing running averages (default: [0.9, 0.999]).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: L2 penalty (default: 0).
        amsgrad: Whether to use AMSGrad variant (default: False).
    """
 
    def __init__(
        self,
        parameters: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        **kwargs
    ):
        """Initialize Adam optimizer.
 
        Args:
            parameters: Model parameters to optimize.
            config: Optimizer configuration dictionary.
            lr: Learning rate.
            betas: Coefficients for computing running averages.
            eps: Term added to denominator for numerical stability.
            weight_decay: L2 penalty.
            amsgrad: Whether to use AMSGrad variant.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(parameters, config)
        self.config.update({
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        })
        self.config.update(kwargs)
 
        self.lr = self.config['lr']
        self.betas = self.config['betas']
        self.eps = self.config['eps']
        self.weight_decay = self.config['weight_decay']
        self.amsgrad = self.config['amsgrad']
 
        self._optimizer = None
        if self.parameters:
            self._build_optimizer()
 
    def _build_optimizer(self):
        """Build the PyTorch Adam optimizer."""
        self._optimizer = TorchAdam(
            self.parameters,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
 
    def set_parameters(self, parameters: List) -> None:
        """Set model parameters to optimize.
 
        Args:
            parameters: Model parameters to optimize.
        """
        self.parameters = list(parameters)
        self._build_optimizer()
 
    def step(self) -> None:
        """Perform one optimization step."""
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call set_parameters() first.")
        self._optimizer.step()
        self._step_count += 1
 
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero out gradients.
 
        Args:
            set_to_none: Whether to set gradients to None instead of zero.
        """
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call set_parameters() first.")
        self._optimizer.zero_grad(set_to_none=set_to_none)
 
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state.
 
        Returns:
            State dictionary.
        """
        return {
            'state': self._optimizer.state_dict() if self._optimizer else {},
            'config': self.config,
            'step_count': self._step_count,
        }
 
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state.
 
        Args:
            state_dict: State dictionary to load.
        """
        if self._optimizer and 'state' in state_dict:
            self._optimizer.load_state_dict(state_dict['state'])
        self.config.update(state_dict.get('config', {}))
        self._step_count = state_dict.get('step_count', 0)
 
    def get_lr(self) -> float:
        """Get current learning rate.
 
        Returns:
            Current learning rate.
        """
        if self._optimizer:
            for param_group in self._optimizer.param_groups:
                return param_group['lr']
        return self.lr
 
    def set_lr(self, lr: float) -> None:
        """Set learning rate.
 
        Args:
            lr: New learning rate.
        """
        self.lr = lr
        self.config['lr'] = lr
        if self._optimizer:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
 
    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, amsgrad={self.amsgrad})"
        )