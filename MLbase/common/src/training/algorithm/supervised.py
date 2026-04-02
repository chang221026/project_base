"""Supervised learning algorithm template."""
from typing import Any, Dict, Optional
 
from .base import BaseAlgorithm, eval_mode
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class SupervisedAlgorithm(BaseAlgorithm):
    """Supervised learning algorithm.
 
    Standard supervised learning with labeled data.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize supervised algorithm.
 
        Args:
            config: Configuration containing:
                - model: Model configuration
                - loss: Loss function configuration
                - optimizer: Optimizer configuration
                - evaluator: Evaluator configuration
        """
        super().__init__(config)
 
    def setup(self) -> None:
        """Setup model, loss, optimizer, and evaluator."""
        # Build model using base class method
        self.model = self._build_model(self.config.get('model', {}))
        logger.info(f"Model created: {type(self.model).__name__}")
 
        # Move model to device
        device = self._get_device()
        self.model = self.model.to(str(device))
        logger.info(f"Model moved to device: {device}")
 
        # Setup loss function using base class method
        self.loss_fn = self._build_loss(self.config.get('loss'))
        logger.info(f"Loss function: {type(self.loss_fn).__name__}")
 
        # Setup optimizer using base class method
        self.optimizer = self._build_optimizer(self.model, self.config.get('optimizer'))
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
 
        # Setup evaluator using base class method
        self.evaluator = self._build_evaluator(self.config.get('evaluator'))
        logger.info(f"Evaluator: {type(self.evaluator).__name__}")
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
 
        Args:
            batch: Tuple of (inputs, targets).
 
        Returns:
            Dictionary with loss value.
        """
        inputs, targets = batch
 
        # Move inputs to device (float32) and targets (keep dtype for labels)
        inputs = self._move_to_device(inputs)
        targets = self._move_to_device(targets, keep_dtype=True)
 
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
 
        # Backward pass
        loss.backward()
        self.optimizer.step()
 
        return {'loss': loss.item()}
 
    def val_step(self, batch: Any) -> Dict[str, float]:
        """Single validation step.
 
        Args:
            batch: Tuple of (inputs, targets).
 
        Returns:
            Dictionary with loss and metrics.
        """
        inputs, targets = batch
 
        # Move inputs to device (float32) and targets (keep dtype for labels)
        inputs = self._move_to_device(inputs)
        targets = self._move_to_device(targets, keep_dtype=True)
 
        # Forward pass
        with eval_mode(self.model):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
 
        # Evaluate
        metrics = self.evaluator.evaluate(outputs, targets)
        metrics['loss'] = loss.item()
 
        return metrics
 
    def predict(self, inputs: Any) -> Any:
        """Make predictions.
 
        Moves inputs to the model's device before inference.
 
        Args:
            inputs: Model inputs (numpy array, tensor, or other format).
 
        Returns:
            Model predictions.
        """
        if not self._trained:
            logger.warning("Model has not been trained yet")
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        # Run inference
        import torch
        with torch.no_grad():
            return self.model(inputs)
 