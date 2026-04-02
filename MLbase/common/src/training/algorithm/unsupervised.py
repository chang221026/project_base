"""Unsupervised learning algorithm template."""
from typing import Any, Dict, Optional
 
from .base import BaseAlgorithm, eval_mode
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class UnsupervisedAlgorithm(BaseAlgorithm):
    """Unsupervised learning algorithm.
 
    Learning without labeled data.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unsupervised algorithm.
 
        Args:
            config: Configuration containing:
                - model: Model configuration
                - loss: Loss function configuration
                - optimizer: Optimizer configuration
        """
        super().__init__(config)
 
    def setup(self) -> None:
        """Setup model, loss, and optimizer."""
        # Build model using base class method
        self.model = self._build_model(self.config.get('model', {}))
        logger.info(f"Model created: {type(self.model).__name__}")
 
        # Move model to device
        device = self._get_device()
        self.model = self.model.to(str(device))
        logger.info(f"Model moved to device: {device}")
 
        # Setup loss function using base class method
        self.loss_fn = self._build_loss(self.config.get('loss', {'type': 'ReconstructionLoss'}))
        logger.info(f"Loss function: {type(self.loss_fn).__name__}")
 
        # Setup optimizer using base class method
        self.optimizer = self._build_optimizer(self.model, self.config.get('optimizer'))
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
 
        Args:
            batch: Input batch (no targets needed).
 
        Returns:
            Dictionary with loss value.
        """
        # Unsupervised learning may use batch directly or extract features
        inputs = self._parse_batch(batch)
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
 
        # Compute unsupervised loss (e.g., reconstruction, clustering)
        loss = self.compute_unsupervised_loss(inputs, outputs)
 
        # Backward pass
        loss.backward()
        self.optimizer.step()
 
        return {'loss': loss.item()}
 
    def val_step(self, batch: Any) -> Dict[str, float]:
        """Single validation step.
 
        Args:
            batch: Input batch.
 
        Returns:
            Dictionary with loss value.
        """
        inputs = self._parse_batch(batch)
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        # Forward pass
        with eval_mode(self.model):
            outputs = self.model(inputs)
            loss = self.compute_unsupervised_loss(inputs, outputs)
 
        return {'loss': loss.item()}
 
    def compute_unsupervised_loss(self, inputs: Any, outputs: Any) -> Any:
        """Compute unsupervised loss.
 
        Args:
            inputs: Input data.
            outputs: Model outputs.
 
        Returns:
            Loss value.
        """
        # Default: use configured loss function
        return self.loss_fn(outputs, inputs)
 
 
class ClusteringAlgorithm(UnsupervisedAlgorithm):
    """Clustering algorithm template."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize clustering algorithm.
 
        Args:
            config: Configuration containing:
                - n_clusters: Number of clusters
                - Other unsupervised config
        """
        super().__init__(config)
        self.n_clusters = self.config.get('n_clusters', 8)
        self.cluster_centers = None
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step with clustering loss."""
        inputs = self._parse_batch(batch)
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        self.optimizer.zero_grad()
 
        # Get embeddings
        embeddings = self.model(inputs)
 
        # Compute clustering loss
        loss = self.compute_clustering_loss(embeddings)
 
        loss.backward()
        self.optimizer.step()
 
        return {'loss': loss.item()}
 
    def compute_clustering_loss(self, embeddings: Any) -> Any:
        """Compute clustering-specific loss.
 
        Args:
            embeddings: Feature embeddings.
 
        Returns:
            Clustering loss.
        """
        # To be implemented by specific clustering algorithms
        raise NotImplementedError("Clustering loss not implemented")