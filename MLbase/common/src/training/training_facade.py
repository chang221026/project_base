"""Training facade module.
 
Provides a unified facade for algorithm management, including
model building, loss function, optimizer, evaluator, training hooks,
and distributed training setup.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
 
from utils.config_management import Config, instantiate
from utils.logger import get_logger
from training.algorithm import ALGORITHMS
from training.hook import HOOKS
from training.distributed.engine import DistributedEngine
 
 
logger = get_logger()
 
 
class TrainingFacade:
    """Facade for algorithm pipeline management.
 
    This class provides a unified interface for all algorithm-related operations:
    - Algorithm building (supervised, reinforcement learning, etc.)
    - Model, loss, optimizer, evaluator management
    - Training hooks management
    - Distributed training setup
    - Training, evaluation, prediction execution
 
    Attributes:
        algorithm: Algorithm instance.
        engine: Distributed engine for distributed training.
    """
 
    def __init__(self, config: Union[str, Path, Dict, Config], logger=None):
        """Initialize training facade from configuration.
 
        Args:
            config: Configuration file path, dictionary, or Config object.
            logger: Optional logger instance. If None, uses default logger.
        """
        self.logger = logger or get_logger()
        self.config = self._load_config(config)
 
        # Initialize training components
        self.algorithm = None
        self.engine: Optional[DistributedEngine] = None
 
        self.logger.info("TrainingFacade initialized")
 
    def _load_config(self, config: Union[str, Path, Dict, Config]) -> Config:
        """Load configuration.
 
        Args:
            config: Configuration in various formats.
 
        Returns:
            Config object.
        """
        from utils.config_management import load_config
 
        if isinstance(config, (str, Path)):
            return load_config(config)
        elif isinstance(config, dict):
            return Config.from_dict(config)
        elif isinstance(config, Config):
            return config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
 
    def setup(self) -> 'TrainingFacade':
        """Setup algorithm from configuration.
 
        Returns:
            Self for chaining.
        """
        algo_config = self.config.get('algorithm', {'type': 'supervised'})
 
        # Build full algorithm configuration - only include non-empty configs
        full_config = {
            'type': algo_config.get('type', 'supervised'),
        }
 
        # Add model config if not empty
        model_config = self.config.get('model')
        if model_config:
            full_config['model'] = model_config
 
        # Add loss config if not empty
        loss_config = self.config.get('loss')
        if loss_config:
            full_config['loss'] = loss_config
 
        # Add optimizer config if not empty
        optimizer_config = self.config.get('optimizer')
        if optimizer_config:
            full_config['optimizer'] = optimizer_config
 
        # Add evaluator config if not empty
        evaluator_config = self.config.get('evaluator')
        if evaluator_config:
            full_config['evaluator'] = evaluator_config
 
        # Add environment config for RL algorithms
        environment_config = self.config.get('environment')
        if environment_config:
            full_config['environment'] = environment_config
 
        # Add actor/critic config for RL algorithms (PPO, SAC, etc.)
        actor_config = self.config.get('actor')
        if actor_config:
            full_config['actor'] = actor_config
 
        critic_config = self.config.get('critic')
        if critic_config:
            full_config['critic'] = critic_config
 
        # Add any additional algorithm-specific config
        for key, value in algo_config.items():
            if key not in full_config:
                full_config[key] = value
 
        # Check if using _target_ for custom algorithm
        if '_target_' in algo_config:
            self.algorithm = instantiate(algo_config)
            algo_type = 'custom'
        else:
            # Build algorithm with config as single argument
            algo_type = full_config.pop('type')
            algo_cls = ALGORITHMS.get(algo_type)
            if algo_cls is None:
                raise ValueError(f"Algorithm '{algo_type}' not found in registry")
            self.algorithm = algo_cls(full_config)
 
        # Setup distributed engine BEFORE creating model
        # This ensures model is created on correct device for each subprocess
        self._setup_distributed()
 
        # Now create model on correct device
        self.algorithm.setup()
        self.logger.info(f"Algorithm built: {algo_type}")
 
        # Setup hooks
        self._setup_hooks()
 
        return self
 
    def _setup_distributed(self) -> None:
        """Setup distributed training engine."""
        from utils.device_management import get_device_manager
 
        dist_config = self.config.get('distributed', {})
        mode = dist_config.get('mode', 'auto')
 
        # Create distributed engine
        self.engine = DistributedEngine()
        self.engine.initialize(auto_setup=(mode == 'auto'))
 
        # Set training mode based on config
        device_manager = get_device_manager()
        training_mode = device_manager.training_mode
 
        # Create strategies
        strategies_config = dist_config.get('strategies', [])
        if strategies_config:
            self.engine.create_strategy_chain(strategies_config)
            self.logger.info(f"Using custom strategies: {[s.get('type') for s in strategies_config]}")
        else:
            self.engine.auto_create_strategies()
 
        self.logger.info(f"Distributed engine initialized for mode: {training_mode.value}")
 
    def _setup_hooks(self) -> None:
        """Setup training hooks from configuration."""
        hooks_config = self.config.get('hooks', {})
 
        for hook_name, hook_cfg in hooks_config.items():
            hook = None
 
            if hook_cfg is None:
                continue
 
            if isinstance(hook_cfg, dict):
                # Check for _target_ first (custom hook)
                if '_target_' in hook_cfg:
                    hook = instantiate(hook_cfg)
                # Check for 'type' field (registered hook)
                elif 'type' in hook_cfg:
                    hook = HOOKS.build(hook_cfg)
                else:
                    # Built-in hook by name
                    hook_cls = HOOKS.get(hook_name)
                    if hook_cls:
                        hook = hook_cls(**hook_cfg)
            else:
                # Assume it's already a hook instance
                hook = hook_cfg
 
            if hook:
                self.algorithm.add_hook(hook)
                self.logger.info(f"Hook added: {hook_name}")
 
    def train(self,
              train_data: Any,
              val_data: Optional[Any] = None,
              **kwargs) -> Dict[str, Any]:
        """Execute training.
 
        Args:
            train_data: Training data loader.
            val_data: Validation data loader (optional).
            **kwargs: Additional arguments.
 
        Returns:
            Training history dictionary.
        """
        if self.algorithm is None:
            raise RuntimeError("Algorithm not initialized. Call setup() first.")
 
        # Prepare data loaders for distributed training
        if self.engine and self.engine.strategies:
            train_data = self.engine.prepare_dataloader(train_data)
            if val_data:
                val_data = self.engine.prepare_dataloader(val_data)
 
            # Prepare model for distributed training
            if self.algorithm.model:
                self.algorithm.model = self.engine.prepare_model(self.algorithm.model)
 
        self.logger.info("Starting training")
        history = self.algorithm.fit(train_data, val_data, **kwargs)
        self.logger.info("Training completed")
        return history
 
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """Execute evaluation.
 
        Args:
            test_data: Test data loader.
 
        Returns:
            Evaluation metrics dictionary.
        """
        if self.algorithm is None:
            raise RuntimeError("Algorithm not initialized. Call setup() first.")
 
        self.logger.info("Starting evaluation")
        metrics = self.algorithm.validate(test_data)
        self.logger.info(f"Evaluation completed: {metrics}")
        return metrics
 
    def predict(self, inputs: Any) -> Any:
        """Make predictions using the trained model.
 
        Args:
            inputs: Input data for prediction.
 
        Returns:
            Model predictions.
        """
        if self.algorithm is None:
            raise RuntimeError("Algorithm not initialized. Call setup() first.")
 
        return self.algorithm.predict(inputs)
 
    def save(self, filepath: str) -> None:
        """Save training facade state.
 
        Args:
            filepath: Path to save state.
        """
        if self.algorithm is None:
            raise RuntimeError("Algorithm not initialized. Call setup() first.")
 
        self.algorithm.save(filepath)
        self.logger.info(f"Training facade state saved to {filepath}")
 
    def load(self, filepath: str) -> None:
        """Load training facade state.
 
        Args:
            filepath: Path to load state from.
        """
        if self.algorithm is None:
            raise RuntimeError("Algorithm not initialized. Call setup() first.")
 
        self.algorithm.load(filepath)
        self.logger.info(f"Training facade state loaded from {filepath}")
 
    def get_algorithm(self):
        """Get algorithm instance.
 
        Returns:
            Algorithm instance or None if not configured.
        """
        return self.algorithm
 
 
def create_training_facade(config: Union[str, Dict, Config]) -> TrainingFacade:
    """Create and setup training facade.
 
    Args:
        config: Configuration file path or dictionary.
 
    Returns:
        Configured TrainingFacade instance.
 
    Example:
        >>> facade = create_training_facade("config.yaml")
        >>> history = facade.train(train_loader, val_loader)
    """
    facade = TrainingFacade(config)
    facade.setup()
    return facade