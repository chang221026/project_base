"""Cross-entropy loss implementation."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
 
from lib.loss_func import LOSSES, BaseLoss
 
 
@LOSSES.register('CrossEntropyLoss')
class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss function.
 
    Computes the cross-entropy loss between predictions and targets.
 
    Config parameters:
        reduction: Reduction method ('mean', 'sum', 'none').
        weight: Optional class weights for handling class imbalance.
        label_smoothing: Label smoothing factor (0.0 for no smoothing).
    """
 
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        """Initialize CrossEntropyLoss.
 
        Args:
            config: Loss configuration dictionary.
            reduction: Reduction method ('mean', 'sum', 'none').
            weight: Optional class weights.
            label_smoothing: Label smoothing factor.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(config)
        self.config.update({
            'reduction': reduction,
            'label_smoothing': label_smoothing,
        })
        self.config.update(kwargs)
 
        self.reduction = self.config['reduction']
        self.label_smoothing = self.config['label_smoothing']
 
        # Register weight as buffer if provided
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
 
        self._loss_fn = nn.CrossEntropyLoss(
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
 
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer tensor.
 
        Args:
            name: Buffer name.
            tensor: Tensor to register.
        """
        setattr(self, name, tensor)
 
    def compute(self, predictions, targets):
        """Compute cross-entropy loss.
 
        Args:
            predictions: Model predictions (logits) of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).
 
        Returns:
            Computed loss value.
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long)
 
        # Ensure targets are long type for CrossEntropyLoss
        targets = targets.long()
 
        return self._loss_fn(predictions, targets)
 
    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}', label_smoothing={self.label_smoothing})"