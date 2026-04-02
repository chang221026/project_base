"""Self-supervised learning algorithm template."""
from typing import Any, Dict, Optional, Callable
 
from .base import BaseAlgorithm, eval_mode
from utils.logger import get_logger


logger = get_logger()
 
 
class SelfSupervisedAlgorithm(BaseAlgorithm):
    """Self-supervised learning algorithm.
 
    Learning from data itself without manual labels.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize self-supervised algorithm.
 
        Args:
            config: Configuration containing:
                - model: Model configuration
                - loss: Loss function configuration
                - optimizer: Optimizer configuration
                - augmentation: Augmentation configuration
        """
        super().__init__(config)
        self.augmentation_fn: Optional[Callable] = None
        self.projection_head = None
 
    def setup(self) -> None:
        """Setup model, loss, optimizer, and augmentation."""
        # Build encoder model using base class method
        self.encoder = self._build_model(self.config.get('encoder', {}))
        self.model = self.encoder
        logger.info(f"Encoder created: {type(self.encoder).__name__}")
 
        # Build projection head (for contrastive learning)
        projection_config = self.config.get('projection_head')
        if projection_config:
            self.projection_head = self._build_model(projection_config)
            logger.info(f"Projection head created: {type(self.projection_head).__name__}")
 
        # Move to device
        device = self._get_device()
        self.encoder = self.encoder.to(str(device))
        if self.projection_head:
            self.projection_head = self.projection_head.to(str(device))
        logger.info(f"Model moved to device: {device}")
 
        # Setup loss function using base class method
        self.loss_fn = self._build_loss(self.config.get('loss', {'type': 'NTXentLoss'}))
        logger.info(f"Loss function: {type(self.loss_fn).__name__}")
 
        # Setup optimizer using base class method
        params = list(self.encoder.parameters())
        if self.projection_head:
            params.extend(self.projection_head.parameters())
        self.optimizer = self._build_optimizer(params, self.config.get('optimizer'))
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
 
        # Setup augmentation
        self.setup_augmentation()
 
    def setup_augmentation(self) -> None:
        """Setup data augmentation for self-supervised learning."""
        aug_config = self.config.get('augmentation', None)
        if aug_config:
            # Build augmentation pipeline
            from utils.registry import Registry
            AUGMENTATIONS = Registry('augmentation')
            self.augmentation_fn = AUGMENTATIONS.build(aug_config)
            logger.info("Augmentation setup complete")
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
 
        Args:
            batch: Input batch.
 
        Returns:
            Dictionary with loss value.
        """
        # Extract inputs
        inputs = self._parse_batch(batch)
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        # Create augmented views
        view1 = self.augment(inputs)
        view2 = self.augment(inputs)
 
        # Forward pass
        self.optimizer.zero_grad()
 
        # Encode
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
 
        # Apply projection head if available
        if self.projection_head:
            z1 = self.projection_head(z1)
            z2 = self.projection_head(z2)
 
        # Compute contrastive loss
        loss = self.loss_fn(z1, z2)
 
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
 
        with eval_mode(self.encoder):
            view1 = self.augment(inputs)
            view2 = self.augment(inputs)
 
            z1 = self.encoder(view1)
            z2 = self.encoder(view2)
 
            if self.projection_head:
                with eval_mode(self.projection_head):
                    z1 = self.projection_head(z1)
                    z2 = self.projection_head(z2)
 
            loss = self.loss_fn(z1, z2)
 
        return {'loss': loss.item()}
 
    def augment(self, inputs: Any) -> Any:
        """Apply augmentation to inputs.
 
        Args:
            inputs: Input data.
 
        Returns:
            Augmented data.
        """
        if self.augmentation_fn:
            return self.augmentation_fn(inputs)
        return inputs
 
    def get_representations(self, inputs: Any) -> Any:
        """Get learned representations (without projection head).
 
        Args:
            inputs: Input data.
 
        Returns:
            Feature representations.
        """
        return self.encoder(inputs)
 
 
class ContrastiveLearning(SelfSupervisedAlgorithm):
    """Contrastive learning (SimCLR style)."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize contrastive learning.
 
        Args:
            config: Configuration including:
                - temperature: Temperature parameter for NT-Xent loss
        """
        super().__init__(config)
        self.temperature = self.config.get('temperature', 0.5)
 
 
class MaskedAutoencoding(SelfSupervisedAlgorithm):
    """Masked autoencoding (MAE style)."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize masked autoencoding.
 
        Args:
            config: Configuration including:
                - mask_ratio: Ratio of inputs to mask
        """
        super().__init__(config)
        self.mask_ratio = self.config.get('mask_ratio', 0.75)
        self.decoder = None
 
    def setup(self) -> None:
        """Setup encoder, decoder, and loss."""
        super().setup()
 
        # Build decoder
        decoder_config = self.config.get('decoder')
        if decoder_config:
            self.decoder = self._build_model(decoder_config)
            device = self._get_device()
            self.decoder = self.decoder.to(str(device))
            logger.info(f"Decoder created: {type(self.decoder).__name__}")
 
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step with masking."""
        inputs = self._parse_batch(batch)
 
        # Move inputs to device
        inputs = self._move_to_device(inputs)
 
        # Mask inputs
        masked_inputs, mask = self.mask_inputs(inputs)
 
        self.optimizer.zero_grad()
 
        # Encode
        latent = self.encoder(masked_inputs)
 
        # Decode
        if self.decoder:
            reconstructed = self.decoder(latent)
        else:
            reconstructed = latent
 
        # Compute reconstruction loss on masked regions
        loss = self.compute_masked_loss(inputs, reconstructed, mask)
 
        loss.backward()
        self.optimizer.step()
 
        return {'loss': loss.item()}
 
    def mask_inputs(self, inputs: Any) -> tuple:
        """Mask random portions of inputs.
 
        Args:
            inputs: Input data.
 
        Returns:
            Tuple of (masked_inputs, mask).
        """
        # To be implemented based on input type
        raise NotImplementedError("Masking not implemented")
 
    def compute_masked_loss(self, original: Any, reconstructed: Any, mask: Any) -> Any:
        """Compute loss on masked regions only.
 
        Args:
            original: Original inputs.
            reconstructed: Reconstructed outputs.
            mask: Mask indicating masked regions.
 
        Returns:
            Loss value.
        """
        # To be implemented
        raise NotImplementedError("Masked loss not implemented")