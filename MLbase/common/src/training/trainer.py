"""Unified training entry point module.
 
Provides a configuration-driven approach to training machine learning models.
"""
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
 
from utils.config_management import Config, load_config, instantiate
from utils.logger import init_logger, get_logger
from utils.device_management import get_device_manager, TrainingMode
from training.distributed.launcher import DistributedLauncher, is_distributed_launched
from training.data_facade import DataFacade
from training.training_facade import TrainingFacade
 
 
class Trainer:
    """Configuration-driven unified training entry point (Facade pattern).
 
    This class acts as a facade that provides a simple, configuration-driven
    interface for training machine learning models. It automatically handles:
    - Configuration loading and validation
    - Data preprocessing pipeline setup (fetcher, analyzer, processors, etc.)
    - Automatic train/val/test split
    - Custom component registration
    - Algorithm building
    - Distributed training setup
    - Hook management
    - Training and evaluation
 
    The facade pattern allows users to train models with minimal code:
 
        # Full pipeline from config - everything is automatic
        trainer = Trainer("config.yaml")
        trainer.train()  # Auto-fetches, processes, splits, and trains
 
    The configuration file can specify the complete data pipeline:
        data:
            fetcher: {type: "CSVDataFetcher", source: "./data.csv"}
            processors: [{type: "StandardScaler"}]
            constructors: [{type: "PolynomialFeatures", degree: 2}]
            selectors: [{type: "VarianceThreshold"}]
            imbalance_handler: {type: "RandomOverSampler"}
        dataset:
            train_ratio: 0.7
            val_ratio: 0.15
            test_ratio: 0.15
 
    Usage:
        # Method 1: Full config-driven (recommended)
        trainer = Trainer("config.yaml")
        trainer.train()  # All data handling is automatic
 
        # Method 2: Override with custom data loaders
        trainer = Trainer("config.yaml")
        trainer.train(train_loader, val_loader)
 
        # Method 3: From dictionary config
        trainer = Trainer({"model": {...}, "data": {...}})
        trainer.train()
    """
 
    def __init__(self, config: Union[str, Path, Dict, Config], _skip_launch_check: bool = False):
        """Initialize trainer from configuration.
 
        Args:
            config: Configuration file path, dictionary, or Config object.
            _skip_launch_check: Internal flag to skip auto-launch check.
                               Used when already launched as subprocess.
 
        Raises:
            ValueError: If config type is not supported.
        """
        # Initialize all instance attributes upfront
        # This ensures attributes exist even if __init__ returns early (e.g., after launching distributed)
        self.data_facade = None
        self.training_facade = None
        self.device_manager = None
        self.logger = None
        self._is_launcher_process = False  # Flag to identify parent process that launched workers
        self._launcher_result: Optional[Dict[str, Any]] = None  # Result collected from distributed training
 
        self._load_and_validate_config(config)
 
        # Check if we need to auto-launch distributed training
        if not _skip_launch_check:
            # For RL algorithms, check if environment is configured
            # If not, disable distributed training (env may be passed to train() as instance)
            algo_config = self.config.get('algorithm', {})
            algo_type = algo_config.get('type', 'supervised')
            rl_algorithms = {'ppo', 'sac', 'a2c', 'ddpg', 'td3'}
 
            if algo_type.lower() in rl_algorithms:
                env_config = self.config.get('environment', {})
                if not env_config:
                    # No environment config - disable distributed training
                    # User may pass env instance to train() which cannot be pickled
                    self.config['distributed'] = {'auto_launch': False}
 
            if self._try_launch_distributed():
                # This is the parent process that launched worker processes
                # Mark it so train() knows to return immediately
                self._is_launcher_process = True
                # Cleanup environment for subsequent Trainer instances
                self._cleanup_after_distributed_launch()
                return
 
        self._setup_logging()
        self._register_custom_components()
        self._setup_data_pipeline()
        self._build_algorithm()
        self._setup_hooks()
 
        self.logger.info("Trainer initialized successfully")
 
    def _try_launch_distributed(self) -> bool:
        """Try to launch distributed training if needed.
 
        Returns:
            True if distributed was launched (parent process should exit).
            False if no launch needed (continue with normal initialization).
        """
        dist_config = self.config.get('distributed', {})
        auto_launch = dist_config.get('auto_launch', True)
 
        if not auto_launch:
            return False
 
        launcher = DistributedLauncher(dist_config)
 
        if launcher.should_launch():
            # Get the config as dict for subprocess
            config_dict = self.config.to_dict()
 
            # Launch distributed training
            # This will spawn multiple processes and block until they complete
            launched = launcher.launch(self._run_in_subprocess, config_dict)
 
            if launched:
                # Collect result from distributed training
                self._launcher_result = launcher.get_result()
 
            return launched
 
        return False
 
    @classmethod
    def _run_in_subprocess(cls, config_dict: Dict) -> Dict[str, Any]:
        """Run training in a subprocess.
 
        This is the entry point for each spawned process.
 
        Args:
            config_dict: Configuration dictionary.
 
        Returns:
            Training history dictionary.
        """
        # Create trainer with skip_launch_check=True to avoid recursive launching
        trainer = cls(config_dict, _skip_launch_check=True)
        return trainer.train()
 
    def _cleanup_after_distributed_launch(self) -> None:
        """Cleanup after distributed training subprocesses have finished.
 
        Resets singleton states and environment variables so that
        subsequent Trainer instances can start fresh.
        """
        import os
        from utils.device_management import DeviceManager
        from utils.distributed_comm import DistributedManager
 
        # Reset singleton states
        DeviceManager.reset()
        DistributedManager.reset()
 
        # Clear distributed environment variables
        env_vars_to_clear = [
            'ML_FRAMEWORK_LAUNCHED', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
            'MASTER_ADDR', 'MASTER_PORT'
        ]
        for var in env_vars_to_clear:
            os.environ.pop(var, None)
 
    def _load_and_validate_config(self, config: Union[str, Path, Dict, Config]) -> None:
        """Load and validate configuration.
 
        Args:
            config: Configuration in various formats.
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
 
        # Apply environment variable overrides
        self.config.apply_env_overrides('ML_')
 
    def _setup_logging(self) -> None:
        """Setup logging system."""
        log_config = self.config.get('logging', {})
        init_logger(
            level=log_config.get('level', 'INFO'),
            log_dir=log_config.get('log_dir', './logs'),
            console_output=log_config.get('console_output', True),
            file_output=log_config.get('file_output', True)
        )
        self.logger = get_logger()
 
    def _register_custom_components(self) -> None:
        """Register custom components from configuration.
 
        Imports modules specified in 'custom_imports' config key.
        These modules should register their components with the
        appropriate registries.
        """
        custom_imports = self.config.get('custom_imports', [])
        for module_path in custom_imports:
            try:
                importlib.import_module(module_path)
                self.logger.info(f"Imported custom module: {module_path}")
            except ImportError as e:
                self.logger.warning(f"Failed to import {module_path}: {e}")
 
    def _setup_data_pipeline(self) -> None:
        """Setup data facade from config.
 
        Creates a DataFacade instance configured from
        the 'data' section of the configuration file.
        """
        self.data_facade = DataFacade(self.config, self.logger)
        self.data_facade.setup()
        self.logger.info("Data facade setup complete")
 
    def _build_algorithm(self) -> None:
        """Build training facade from configuration."""
        self.training_facade = TrainingFacade(self.config, self.logger)
        self.training_facade.setup()
        self.logger.info("Training facade setup complete")
 
        # Distributed training is now handled by TrainingFacade internally
        # No need to call _setup_distributed() separately
 
    def _setup_hooks(self) -> None:
        """Setup training hooks.
 
        Note: Hooks are now managed by TrainingFacade during its setup.
        This method is kept for backward compatibility but is a no-op.
        """
        # Hooks are now managed by TrainingFacade during setup()
        # This method kept for backward compatibility
        pass
 
    def _build_data_from_config(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Build datasets from configuration.
 
        Delegates to DataFacade for data loading concerns.
 
        Returns:
            Tuple of (train_dataset, val_dataset) or (None, None) if not configured.
        """
        if self.data_facade:
            train_dataset, val_dataset, _ = self.data_facade.get_data_loaders()
            return train_dataset, val_dataset
        return None, None
 
    def _is_rl_algorithm(self) -> bool:
        """Check if the algorithm is a reinforcement learning algorithm.
 
        Returns:
            True if RL algorithm.
        """
        from training.algorithm import RLAlgorithm
        algorithm = self.training_facade.get_algorithm() if self.training_facade else None
        return algorithm is not None and isinstance(algorithm, RLAlgorithm)
 
    def train(self,
              train_data: Optional[Any] = None,
              val_data: Optional[Any] = None,
              **kwargs) -> Dict[str, Any]:
        """Execute training.
 
        Args:
            train_data: Training dataset. If None, builds from config.
            val_data: Validation dataset. If None, builds from config.
            **kwargs: Additional arguments for RL training (total_steps, eval_freq, eval_episodes, env).
 
        Returns:
            Training history dictionary.
 
        Raises:
            ValueError: If no training data provided for supervised learning.
        """
        # If this is the launcher process (parent), return collected results
        # The actual training happens in spawned subprocesses
        if self._is_launcher_process:
            # Return the result collected from rank 0 process
            if self._launcher_result is not None:
                return self._launcher_result
            return {'train': [], 'val': []}
 
        # Check if this is an RL algorithm
        if self._is_rl_algorithm():
            return self._train_rl(**kwargs)
        else:
            return self._train_supervised(train_data, val_data)
 
    def _train_rl(self, **kwargs) -> Dict[str, Any]:
        """Execute RL training.
 
        Args:
            **kwargs: RL training arguments (total_steps, eval_freq, eval_episodes, env).
 
        Returns:
            Training history dictionary.
        """
        # Get training parameters from config or kwargs
        training_config = self.config.get('training', {})
        total_steps = kwargs.get('total_steps', training_config.get('total_steps', 10000))
        eval_freq = kwargs.get('eval_freq', training_config.get('eval_freq', 1000))
        eval_episodes = kwargs.get('eval_episodes', training_config.get('eval_episodes', 10))
        env = kwargs.get('env')  # Optional environment instance (None if using config)
 
        # Execute RL training via TrainingFacade
        self.logger.info(f"Starting RL training for {total_steps} steps")
        algorithm = self.training_facade.get_algorithm()
        history = algorithm.fit(
            env=env,
            total_steps=total_steps,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes
        )
 
        self.logger.info("RL training completed")
        return history
 
    def _train_supervised(self,
                          train_data: Optional[Any],
                          val_data: Optional[Any]) -> Dict[str, Any]:
        """Execute supervised learning training.
 
        Args:
            train_data: Training data loader. If None, builds from config.
            val_data: Validation data loader. If None, builds from config.
 
        Returns:
            Training history dictionary.
        """
        # Build data from config if not provided
        if train_data is None:
            train_data, val_data = self._build_data_from_config()
            if train_data is None:
                raise ValueError(
                    "No training data provided. Either pass train_data argument "
                    "or configure 'data' section in config with fetcher or path."
                )
 
        # Distributed training is now handled by TrainingFacade internally
        # TrainingFacade.train() handles data loader and model preparation
 
        # Get training parameters
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 10)
 
        # Execute training via TrainingFacade
        self.logger.info(f"Starting training for {epochs} epochs")
        history = self.training_facade.train(train_data, val_data, epochs=epochs)
 
        self.logger.info("Training completed")
        return history
 
    def evaluate(self, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """Execute evaluation.
 
        Args:
            test_data: Test dataset. If None, builds from config.
 
        Returns:
            Evaluation metrics dictionary.
        """
        if test_data is None and self.data_facade:
            _, _, test_data = self.data_facade.get_data_loaders()
 
        if test_data is None:
            raise ValueError("No test data provided for evaluation")
 
        self.logger.info("Starting evaluation")
        metrics = self.training_facade.evaluate(test_data)
        self.logger.info(f"Evaluation completed: {metrics}")
 
        return metrics
 
    def predict(self, inputs: Any) -> Any:
        """Make predictions using the trained model.
 
        Args:
            inputs: Input data for prediction.
 
        Returns:
            Model predictions.
        """
        # Skip if this is the launcher process (no algorithm initialized)
        if self._is_launcher_process:
            return None
        return self.training_facade.predict(inputs)
 
    def save(self, filepath: str) -> None:
        """Save trainer state.
 
        Args:
            filepath: Path to save state.
        """
        # Skip if this is the launcher process (no algorithm initialized)
        if self._is_launcher_process:
            return
        self.training_facade.save(filepath)
        self.logger.info(f"Trainer state saved to {filepath}")
 
    def load(self, filepath: str) -> None:
        """Load trainer state.
 
        Args:
            filepath: Path to load state from.
        """
        # Skip if this is the launcher process (no algorithm initialized)
        if self._is_launcher_process:
            return
        self.training_facade.load(filepath)
        self.logger.info(f"Trainer state loaded from {filepath}")
 
    def get_config(self) -> Dict[str, Any]:
        """Get trainer configuration.
 
        Returns:
            Configuration dictionary.
        """
        return self.config.to_dict()
 
 
def train(config: Union[str, Dict, Config],
          train_data: Optional[Any] = None,
          val_data: Optional[Any] = None) -> Dict[str, Any]:
    """One-line training function.
 
    This is a convenience function that creates a Trainer and runs training.
 
    Args:
        config: Configuration file path, dictionary, or Config object.
        train_data: Training dataset (optional).
        val_data: Validation dataset (optional).
 
    Returns:
        Training history dictionary.
 
    Example:
        >>> # Train from config file
        >>> history = train("config.yaml")
 
        >>> # Train with passed datasets
        >>> history = train("config.yaml", train_dataset, val_dataset)
 
        >>> # Train from dictionary config
        >>> config = {"model": {"type": "SimpleModel"}, "training": {"epochs": 10}}
        >>> history = train(config)
    """
    trainer = Trainer(config)
    return trainer.train(train_data, val_data)