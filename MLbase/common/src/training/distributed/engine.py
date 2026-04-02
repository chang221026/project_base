"""Distributed training engine.
 
Provides unified interface for distributed training with automatic
device and strategy management.
"""
from typing import Any, Dict, List, Optional, Callable
 
from utils.distributed_comm import DistributedManager, get_dist_manager
from utils.device_management import (
    get_device_manager, TrainingMode, DeviceConfig
)
from utils.logger import get_logger
from .strategy import (
    ParallelStrategy,
    DataParallelStrategy,
    DistributedDataParallelStrategy,
    ModelParallelStrategy,
    PipelineParallelStrategy,
    FSDPStrategy
)
 
 
logger = get_logger()
 
 
class DistributedEngine:
    """Distributed training engine.
 
    Manages distributed environment and applies parallel strategies.
    """
 
    STRATEGIES = {
        'data_parallel': DataParallelStrategy,
        'ddp': DistributedDataParallelStrategy,
        'distributed_data_parallel': DistributedDataParallelStrategy,
        'model_parallel': ModelParallelStrategy,
        'pipeline_parallel': PipelineParallelStrategy,
        'fsdp': FSDPStrategy,
    }
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize distributed engine.
 
        Args:
            config: Engine configuration containing:
                - backend: Distributed backend
                - init_method: Initialization method
                - world_size: Total number of processes
                - rank: Global rank
                - local_rank: Local rank
                - strategies: List of strategy configurations
        """
        self.config = config or {}
        self.dist_manager = get_dist_manager()
        self.device_manager = get_device_manager()
        self.strategies: List[ParallelStrategy] = []
        self._initialized = False
 
    def initialize(self,
                   backend: Optional[str] = None,
                   init_method: Optional[str] = None,
                   world_size: Optional[int] = None,
                   rank: Optional[int] = None,
                   auto_setup: bool = True) -> None:
        """Initialize distributed environment.
 
        Args:
            backend: Communication backend.
            init_method: Initialization method.
            world_size: Total number of processes.
            rank: Global rank.
            auto_setup: If True, automatically detect mode and setup devices.
                       If False, use manually configured training mode.
        """
        if self._initialized:
            logger.warning("Distributed engine already initialized")
            return
 
        if auto_setup:
            # Automatically detect and setup training mode
            self.device_manager.setup_for_distributed()
            mode = self.device_manager.training_mode
 
            logger.info(f"Auto-detected training mode: {mode.value}")
 
            # Initialize distributed manager based on training mode
            self.dist_manager.init_from_device_manager(self.device_manager)
 
            # Log device configuration
            config = self.device_manager.get_device_config()
            logger.info(
                f"Device configuration: mode={config.mode.value}, "
                f"device_ids={config.device_ids}, local_rank={config.local_rank}"
            )
        else:
            # Manual initialization using existing logic
            self.dist_manager.init_distributed(
                backend=backend,
                init_method=init_method
            )
 
            # Set device for current process
            if self.dist_manager.is_distributed():
                local_rank = self.dist_manager.get_local_rank()
                # Get current device type to build correct device string (support NPU/CUDA/CPU)
                device = self.device_manager.get_current_device()
                if device:
                    device_str = f"{device.type.value}:{local_rank}"
                else:
                    # Fallback to npu
                    device_str = f"npu:{local_rank}"
                self.device_manager.set_device(device_str)
                logger.info(f"Process {rank} using device {device_str}")
 
        self._initialized = True
        logger.info("Distributed engine initialized")
 
    def initialize_with_config(self, config: DeviceConfig) -> None:
        """Initialize distributed environment with explicit configuration.
 
        Args:
            config: DeviceConfig specifying training mode and devices.
        """
        if self._initialized:
            logger.warning("Distributed engine already initialized")
            return
 
        # Set training mode from config
        self.device_manager.set_training_mode(
            config.mode,
            device_ids=config.device_ids if config.device_ids else None
        )
 
        # Initialize distributed manager
        self.dist_manager.init_from_device_manager(self.device_manager)
 
        self._initialized = True
        logger.info(f"Distributed engine initialized with config: {config}")
 
    def auto_create_strategies(self) -> None:
        """Automatically create strategy chain based on current training mode.
 
        Creates appropriate strategy based on training mode:
        - SINGLE: No wrapping (single device)
        - SINGLE_MACHINE_MULTI_DEVICE: DDP strategy (not DataParallel)
        - MULTI_MACHINE_MULTI_DEVICE: DDP strategy
 
        Important:
            For SINGLE_MACHINE_MULTI_DEVICE mode, DDP initialization requires that
            the training script be launched via `torchrun`. The launcher sets up
            the necessary environment variables (WORLD_SIZE, RANK, LOCAL_RANK,
            MASTER_ADDR, MASTER_PORT) for each process.
 
            Example:
                torchrun --nproc_per_node=<device_count> --master_port=29500 train.py
 
            If not launched via `torchrun`, DDP will not be initialized despite
            detecting multiple devices. See `DistributedManager.init_from_device_manager`
            for more details.
        """
        mode = self.device_manager.training_mode
        device_ids = self.device_manager.device_ids
 
        if mode == TrainingMode.SINGLE:
            # No strategy needed for single device
            self.strategies = []
            logger.info("Single device mode - no parallel strategy created")
 
        elif mode == TrainingMode.SINGLE_MACHINE_MULTI_DEVICE:
            # Use DDP for single machine multi-device (not DataParallel)
            # Initialize distributed for DDP
            self.dist_manager.init_from_device_manager(self.device_manager)
            self.create_strategy_chain([{
                'type': 'ddp',
                'find_unused_parameters': False
            }])
            logger.info(f"DDP strategy created for single machine devices: {device_ids}")
 
        elif mode == TrainingMode.MULTI_MACHINE_MULTI_DEVICE:
            # DDP for multi-machine multi-device
            self.create_strategy_chain([{
                'type': 'ddp',
                'find_unused_parameters': False
            }])
            logger.info("DDP strategy created for distributed training")
 
    def create_strategy_chain(self, strategy_configs: List[Dict[str, Any]]) -> None:
        """Create chain of parallel strategies.
 
        Args:
            strategy_configs: List of strategy configurations.
                             Each should have 'type' and other parameters.
        """
        self.strategies = []
 
        for config in strategy_configs:
            strategy_type = config.get('type', 'data_parallel')
 
            if strategy_type not in self.STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy_type}")
 
            strategy_class = self.STRATEGIES[strategy_type]
            strategy = strategy_class(config)
            self.strategies.append(strategy)
 
            logger.info(f"Added strategy: {strategy_type}")
 
    def prepare_model(self, model: Any) -> Any:
        """Prepare model with all strategies.
 
        Args:
            model: Model to prepare.
 
        Returns:
            Prepared model.
        """
        for strategy in self.strategies:
            model = strategy.prepare_model(model)
        return model
 
    def prepare_optimizer(self, optimizer: Any) -> Any:
        """Prepare optimizer with all strategies.
 
        Args:
            optimizer: Optimizer to prepare.
 
        Returns:
            Prepared optimizer.
        """
        for strategy in self.strategies:
            optimizer = strategy.prepare_optimizer(optimizer)
        return optimizer
 
    def prepare_dataloader(self, dataloader: Any) -> Any:
        """Prepare dataloader with all strategies.
 
        Args:
            dataloader: DataLoader to prepare.
 
        Returns:
            Prepared DataLoader.
        """
        for strategy in self.strategies:
            dataloader = strategy.prepare_dataloader(dataloader)
        return dataloader
 
    def wrap_training_step(self, train_step_fn: Callable) -> Callable:
        """Wrap training step for distributed training.
 
        Args:
            train_step_fn: Original training step function.
 
        Returns:
            Wrapped training step function.
        """
        def wrapped_step(batch):
            # Unpack batch based on distributed requirements
            if self.dist_manager.is_distributed():
                # Synchronize batch across processes if needed
                pass
 
            # Call original training step
            metrics = train_step_fn(batch)
 
            # All-reduce metrics for logging
            if self.dist_manager.is_distributed():
                metrics = self._all_reduce_metrics(metrics)
 
            return metrics
 
        return wrapped_step
 
    def _all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce metrics across processes.
 
        Args:
            metrics: Metrics dictionary.
 
        Returns:
            Averaged metrics.
        """
        try:
            import torch
 
            reduced_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tensor = torch.tensor(value)
                    self.dist_manager.all_reduce(tensor, op='mean')
                    reduced_metrics[key] = tensor.item()
                else:
                    reduced_metrics[key] = value
 
            return reduced_metrics
        except ImportError:
            return metrics
 
    def is_main_process(self) -> bool:
        """Check if current process is main process.
 
        Returns:
            True if main process.
        """
        return self.dist_manager.is_main_process()
 
    def get_rank(self) -> int:
        """Get current process rank.
 
        Returns:
            Process rank.
        """
        return self.dist_manager.get_rank()
 
    def get_world_size(self) -> int:
        """Get total number of processes.
 
        Returns:
            World size.
        """
        return self.dist_manager.get_world_size()
 
    def barrier(self) -> None:
        """Synchronization barrier."""
        self.dist_manager.barrier()
 
    def broadcast(self, tensor: Any, src: int = 0) -> None:
        """Broadcast tensor from source.
 
        Args:
            tensor: Tensor to broadcast.
            src: Source rank.
        """
        self.dist_manager.broadcast(tensor, src)
 
    def all_gather(self, tensor: Any) -> List[Any]:
        """All-gather operation.
 
        Args:
            tensor: Tensor to gather.
 
        Returns:
            List of tensors from all processes.
        """
        return self.dist_manager.all_gather(tensor)
 
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        self.dist_manager.destroy()
        self._initialized = False
        logger.info("Distributed engine cleanup complete")
 
 
class DistributedTrainer:
    """Distributed trainer with automatic batch distribution."""
 
    def __init__(self, algorithm, engine: Optional[DistributedEngine] = None):
        """Initialize distributed trainer.
 
        Args:
            algorithm: Training algorithm.
            engine: Distributed engine (creates default if None).
        """
        self.algorithm = algorithm
        self.engine = engine or DistributedEngine()
        self._setup = False
 
    def setup(self) -> None:
        """Setup distributed training."""
        if self._setup:
            return
 
        # Initialize distributed
        self.engine.initialize()
 
        # Prepare model
        if self.algorithm.model:
            self.algorithm.model = self.engine.prepare_model(self.algorithm.model)
 
        # Prepare optimizer
        if self.algorithm.optimizer:
            self.algorithm.optimizer = self.engine.prepare_optimizer(
                self.algorithm.optimizer
            )
 
        self._setup = True
        logger.info("Distributed trainer setup complete")
 
    def fit(self, train_loader, val_loader=None, epochs: int = 10) -> Dict[str, Any]:
        """Train with distributed support.
 
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs.
 
        Returns:
            Training history.
        """
        self.setup()
 
        # Prepare dataloaders
        train_loader = self.engine.prepare_dataloader(train_loader)
        if val_loader:
            val_loader = self.engine.prepare_dataloader(val_loader)
 
        # Run training
        history = self.algorithm.fit(train_loader, val_loader, epochs)
 
        return history
 
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.engine.cleanup()