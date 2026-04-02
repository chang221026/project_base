"""Distributed training strategies.
 
Provides various parallel strategies for distributed training.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
 
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class ParallelStrategy(ABC):
    """Base class for parallel strategies."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy.
 
        Args:
            config: Strategy configuration.
        """
        self.config = config or {}
 
    @abstractmethod
    def prepare_model(self, model):
        """Prepare model for distributed training.
 
        Args:
            model: Model to parallelize.
 
        Returns:
            Parallelized model.
        """
        pass
 
    def prepare_optimizer(self, optimizer):
        """Prepare optimizer for distributed training.
 
        Default implementation returns optimizer unchanged.
 
        Args:
            optimizer: Optimizer to wrap.
 
        Returns:
            Wrapped optimizer.
        """
        return optimizer
 
    def prepare_dataloader(self, dataloader):
        """Prepare dataloader for distributed training.
 
        Default implementation returns dataloader unchanged.
 
        Args:
            dataloader: DataLoader to wrap.
 
        Returns:
            Wrapped DataLoader.
        """
        return dataloader
 
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.config.copy()
 
 
class DataParallelStrategy(ParallelStrategy):
    """Data Parallel strategy.
 
    Replicates model on each device, splits data across devices.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data parallel strategy."""
        super().__init__(config)
        self.device_ids = self.config.get('device_ids', None)
 
    def prepare_model(self, model):
        """Wrap model for data parallel."""
        try:
            import torch.nn as nn
            if self.device_ids:
                model = nn.DataParallel(model, device_ids=self.device_ids)
            else:
                model = nn.DataParallel(model)
            logger.info("Model wrapped with DataParallel")
        except ImportError:
            logger.warning("torch not available for DataParallel")
        return model
 
 
class DistributedDataParallelStrategy(ParallelStrategy):
    """Distributed Data Parallel strategy.
 
    Each process works on a subset of data with synchronized gradients.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DDP strategy."""
        super().__init__(config)
        self.find_unused_parameters = self.config.get('find_unused_parameters', False)
 
    def prepare_model(self, model):
        """Wrap model with DDP.
 
        Moves the model to the correct device before wrapping with DDP.
        For NPU/CUDA, sets the current device first to ensure proper DDP initialization.
        """
        try:
            import torch
            from torch.nn.parallel import DistributedDataParallel as DDP
            from utils.device_management import DeviceType, get_device_manager
 
            if not torch.distributed.is_initialized():
                return model
 
            device_manager = get_device_manager()
            device = device_manager.get_current_device()
            device_type = device.type
            device_index = device.index
 
            # Set device and move model
            self._set_device(device_type, device_index)
            model = model.to(device.name)
            logger.info(f"Model moved to device: {device.name}")
 
            # Wrap with DDP
            model = self._wrap_ddp(model, device_type, device_index)
            logger.info("Model wrapped with DistributedDataParallel")
 
        except ImportError as e:
            logger.warning(f"torch not available for DDP: {e}")
        return model
 
    def _set_device(self, device_type, device_index: int) -> None:
        """Set current device for NPU or CUDA."""
        import torch
        from utils.device_management import DeviceType
        if device_type == DeviceType.NPU:
            import torch_npu
            torch_npu.npu.set_device(device_index)
        elif device_type == DeviceType.CUDA:
            torch.cuda.set_device(device_index)
        logger.info(f"Set {device_type.value} device: {device_index}")
 
    def _wrap_ddp(self, model, device_type, device_index: int):
        """Wrap model with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        from utils.device_management import DeviceType
 
        if device_type == DeviceType.CPU:
            logger.warning("DDP on CPU is not recommended for performance")
            return DDP(model, find_unused_parameters=self.find_unused_parameters)
 
        return DDP(
            model,
            device_ids=[device_index],
            output_device=device_index if device_type == DeviceType.CUDA else None,
            find_unused_parameters=self.find_unused_parameters
        )
 
    def prepare_dataloader(self, dataloader):
        """Wrap dataloader/dataset with distributed sampler.
 
        Supports both DataLoader and Dataset inputs.
        For Dataset, creates a simple wrapper with sampler support.
        """
        try:
            import torch
            from torch.utils.data.distributed import DistributedSampler
 
            # Check if input is a Dataset (not DataLoader)
            from training.data.dataset_building import Dataset
            if isinstance(dataloader, Dataset):
                # Always wrap Dataset with DatasetWrapper for consistency
                class DatasetWrapper:
                    """Simple wrapper for Dataset with sampler support."""
                    def __init__(self, dataset):
                        self.dataset = dataset
                        self.sampler = None
                        self.batch_size = 32
 
                    def __iter__(self):
                        indices = list(range(len(self.dataset)))
                        if self.sampler is not None:
                            # Use sampler indices if available
                            indices = list(self.sampler)
 
                        for i in range(0, len(indices), self.batch_size):
                            batch_indices = indices[i:i + self.batch_size]
                            batch = [self.dataset[idx] for idx in batch_indices]
                            samples = [item[0] for item in batch]
                            targets = [item[1] for item in batch]
                            yield samples, targets
 
                    def __len__(self):
                        return len(self.dataset)
 
                dataloader = DatasetWrapper(dataloader)
                logger.info("Dataset wrapped with DatasetWrapper")
 
                # If distributed is initialized, add DistributedSampler
                if torch.distributed.is_initialized():
                    sampler = DistributedSampler(
                        dataloader.dataset,
                        num_replicas=torch.distributed.get_world_size(),
                        rank=torch.distributed.get_rank()
                    )
                    dataloader.sampler = sampler
                    logger.info("Dataset wrapped with DistributedSampler")
 
        except ImportError:
            logger.warning("torch not available for distributed sampler")
        except Exception as e:
            logger.warning(f"Failed to wrap dataloader: {e}")
 
        return dataloader
 
 
class ModelParallelStrategy(ParallelStrategy):
    """Model Parallel strategy.
 
    Splits model layers across devices.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model parallel strategy."""
        super().__init__(config)
        self.layer_devices = self.config.get('layer_devices', {})
 
    def prepare_model(self, model):
        """Split model across devices."""
        for layer_name, device in self.layer_devices.items():
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                layer.to(device)
                logger.info(f"Layer {layer_name} moved to {device}")
        return model
 
 
class PipelineParallelStrategy(ParallelStrategy):
    """Pipeline Parallel strategy.
 
    Splits model into stages processed sequentially.
 
    Note: This is a placeholder implementation. Full pipeline parallelism
    requires torch.distributed.pipeline or similar framework support.
    """
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline parallel strategy."""
        super().__init__(config)
        self.num_stages = self.config.get('num_stages', 2)
        self.chunks = self.config.get('chunks', 1)
 
    def prepare_model(self, model):
        """Setup model for pipeline parallelism."""
        logger.info(f"Model prepared for {self.num_stages}-stage pipeline")
        return model
 
 
class FSDPStrategy(ParallelStrategy):
    """Fully Sharded Data Parallel strategy (PyTorch FSDP)."""
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FSDP strategy."""
        super().__init__(config)
        self.cpu_offload = self.config.get('cpu_offload', False)
        self.mixed_precision = self.config.get('mixed_precision', False)
 
    def prepare_model(self, model):
        """Wrap model with FSDP."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import default_auto_wrap_policy
 
            model = FSDP(
                model,
                auto_wrap_policy=default_auto_wrap_policy,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision
            )
            logger.info("Model wrapped with FSDP")
        except ImportError:
            logger.warning("FSDP not available in this PyTorch version")
        return model