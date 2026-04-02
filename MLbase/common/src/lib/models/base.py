"""Base model class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
 
import torch.nn as nn
 
 
class BaseModel(nn.Module, ABC):
    """Base class for all models, inheriting from nn.Module and ABC.
 
    All models should inherit from this class and implement
    the required abstract methods.
 
    By inheriting from nn.Module, models automatically get:
    - named_parameters(), parameters()
    - modules()
    - to(), train(), eval()
    - state_dict(), load_state_dict()
    - Proper DDP wrapping support
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model.
 
        Args:
            config: Model configuration dictionary.
        """
        super().__init__()  # Initialize nn.Module
        self.config = config or {}
        self._built = False
 
    @abstractmethod
    def forward(self, inputs):
        """Forward pass.
 
        Args:
            inputs: Model inputs.
 
        Returns:
            Model outputs.
        """
        pass
 
    @abstractmethod
    def build(self, input_shape=None):
        """Build model.
 
        Args:
            input_shape: Input shape for building the model.
 
        Returns:
            Self for chaining.
        """
        self._built = True
        return self
 
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
 
    def save(self, filepath: str) -> None:
        """Save model to file.
 
        Args:
            filepath: Path to save model.
        """
        raise NotImplementedError("Save not implemented")
 
    @classmethod
    def load(cls, filepath: str):
        """Load model from file.
 
        Args:
            filepath: Path to load model from.
 
        Returns:
            Loaded model instance.
        """
        raise NotImplementedError("Load not implemented")