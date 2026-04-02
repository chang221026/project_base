"""MLP (Multi-Layer Perceptron) model implementation."""
from typing import Any, Dict, List, Optional
 
import torch
import torch.nn as nn
 
from lib.models import MODELS, BaseModel
 
 
@MODELS.register('MLP')
class MLP(BaseModel):
    """Multi-Layer Perceptron model.
 
    A simple feedforward neural network with configurable hidden layers.
 
    Config parameters:
        input_dim: Input feature dimension.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Output dimension (number of classes for classification).
        activation: Activation function name ('relu', 'tanh', 'sigmoid', 'gelu').
    """
 
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'gelu': nn.GELU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
    }
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 10,
        activation: str = 'relu',
        **kwargs
    ):
        """Initialize MLP model.
 
        Args:
            config: Model configuration dictionary.
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output dimension.
            activation: Activation function name.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'input_dim': input_dim,
            'hidden_dims': hidden_dims or [256, 128],
            'output_dim': output_dim,
            'activation': activation,
        })
        self.config.update(kwargs)
 
        self.input_dim = self.config['input_dim']
        self.hidden_dims = self.config['hidden_dims']
        self.output_dim = self.config['output_dim']
        self.activation_name = self.config['activation']
 
        # Build network immediately so nn.Module.parameters() works
        self._build_network()
        self._built = True
 
    def _build_network(self):
        """Build the neural network layers."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
 
        activation_cls = self.ACTIVATIONS.get(self.activation_name, nn.ReLU)
 
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(activation_cls())
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.Dropout(0.1))
 
        self.network = nn.Sequential(*layers)
 
    def build(self, input_shape=None):
        """Build model (network is already built in __init__).
 
        This method is kept for API compatibility and can update input_dim.
 
        Args:
            input_shape: Input shape for updating input dimension.
 
        Returns:
            Self for chaining.
        """
        if input_shape is not None:
            new_input_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape
            if new_input_dim != self.input_dim:
                # Rebuild network with new input dimension
                self.input_dim = new_input_dim
                self.config['input_dim'] = self.input_dim
                self._build_network()
        return self
 
    def forward(self, inputs):
        """Forward pass.
 
        Args:
            inputs: Input tensor of shape (batch_size, input_dim) or (input_dim,).
 
        Returns:
            Output tensor of shape (batch_size, output_dim) or (output_dim,).
        """
        if isinstance(inputs, torch.Tensor):
            x = inputs
        else:
            x = torch.tensor(inputs, dtype=torch.float32)
 
        # Handle 1D input (single sample) - required for BatchNorm1d
        # BatchNorm requires batch_size > 1 in training mode, so we need
        # to temporarily switch to eval mode for single sample inference
        squeeze_output = False
        use_eval_mode = False
 
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
            use_eval_mode = True
        elif x.dim() == 2 and x.size(0) == 1:
            # batch_size == 1 also needs eval mode for BatchNorm
            use_eval_mode = True
 
        if use_eval_mode and self.training:
            # Temporarily switch to eval mode for BatchNorm
            was_training = True
            self.network.eval()
        else:
            was_training = False
 
        try:
            output = self.network(x)
        finally:
            if was_training:
                self.network.train()
 
        if squeeze_output:
            output = output.squeeze(0)  # Remove batch dimension for single sample
 
        return output
 
    def save(self, filepath: str) -> None:
        """Save model to file.
 
        Args:
            filepath: Path to save model.
        """
        torch.save({
            'state_dict': self.network.state_dict(),
            'config': self.config,
        }, filepath)
 
    @classmethod
    def load(cls, filepath: str):
        """Load model from file.
 
        Args:
            filepath: Path to load model from.
 
        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(filepath)
        model = cls(config=checkpoint['config'])
        if checkpoint['state_dict']:
            model.network.load_state_dict(checkpoint['state_dict'])
        return model