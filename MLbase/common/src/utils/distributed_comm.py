"""Distributed communication module.
 
Provides unified interface for distributed training communication.
Supports both PyTorch DDP and custom backends.
"""
import os
from typing import Optional, Callable, Any, List
from enum import Enum
 
from .exception import DistributedInitError, DistributedCommunicationError
from .logger import get_logger
 
 
logger = get_logger()
 
 
class Backend(Enum):
    """Distributed backend types."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    HCCL = "hccl"  # Huawei Collective Communication Library for Ascend NPU
 
 
class DistributedManager:
    """Manages distributed training environment.
 
    Singleton pattern for consistent distributed state.
    """
 
    _instance: Optional['DistributedManager'] = None
    _initialized: bool = False
 
    def __new__(cls, *args, **kwargs):
        """Ensure singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
 
    def __init__(self,
                 backend: Optional[Backend] = None,
                 init_method: Optional[str] = None,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 local_rank: Optional[int] = None):
        """Initialize distributed manager.
 
        Args:
            backend: Communication backend.
            init_method: Initialization method URL.
            world_size: Total number of processes.
            rank: Global rank of current process.
            local_rank: Local rank on current node.
        """
        if DistributedManager._initialized:
            return
 
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self._is_distributed = False
        self._process_group = None
 
        DistributedManager._initialized = True
 
    @classmethod
    def reset(cls) -> None:
        """Reset singleton state for testing or re-initialization."""
        cls._instance = None
        cls._initialized = False
 
    def _is_npu_available(self) -> bool:
        """Check if NPU is available.
 
        Returns:
            True if torch_npu is available and NPU devices exist.
        """
        try:
            import torch_npu
            return torch_npu.npu.is_available()
        except ImportError:
            return False
 
    def init_distributed(self,
                         backend: Optional[Backend] = None,
                         init_method: Optional[str] = None) -> None:
        """Initialize distributed environment.
 
        Args:
            backend: Communication backend. Auto-detected if None.
            init_method: Initialization method. Auto-detected if None.
 
        Raises:
            DistributedInitError: If initialization fails.
        """
        if self._is_distributed:
            logger.warning("Distributed already initialized")
            return
 
        try:
            import torch.distributed as dist
            import torch
 
            # Auto-detect backend (Priority: NPU (HCCL) -> CUDA (NCCL) -> GLOO)
            if backend is None:
                if self._is_npu_available():
                    backend = Backend.HCCL
                elif torch.cuda.is_available():
                    backend = Backend.NCCL
                else:
                    backend = Backend.GLOO
 
            # Auto-detect init method
            if init_method is None:
                if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                    init_method = "env://"
                else:
                    init_method = "file:///tmp/sharedfile"
 
            # Get world size and rank from environment or parameters
            world_size = self.world_size or int(os.environ.get("WORLD_SIZE", 1))
            rank = self.rank or int(os.environ.get("RANK", 0))
            local_rank = self.local_rank or int(os.environ.get("LOCAL_RANK", 0))
 
            if world_size > 1:
                dist.init_process_group(
                    backend=backend.value,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank
                )
 
                self._is_distributed = True
                self.backend = backend
                self.init_method = init_method
                self.world_size = world_size
                self.rank = rank
                self.local_rank = local_rank
                self._process_group = dist.group.WORLD
 
                logger.info(
                    f"Distributed initialized: rank={rank}/{world_size}, "
                    f"backend={backend.value}"
                )
            else:
                logger.info("Single process mode, distributed not initialized")
 
        except Exception as e:
            raise DistributedInitError(f"Failed to initialize distributed: {e}")
 
    def is_distributed(self) -> bool:
        """Check if running in distributed mode.
 
        Returns:
            True if distributed is initialized.
        """
        return self._is_distributed
 
    def is_main_process(self) -> bool:
        """Check if current process is main process.
 
        Returns:
            True if this is the main process.
        """
        return not self._is_distributed or self.rank == 0
 
    def get_rank(self) -> int:
        """Get current process rank.
 
        Returns:
            Process rank (0 if not distributed).
        """
        return self.rank if self._is_distributed and self.rank is not None else 0
 
    def get_world_size(self) -> int:
        """Get total number of processes.
 
        Returns:
            World size (1 if not distributed).
        """
        return self.world_size if self._is_distributed and self.world_size is not None else 1
 
    def get_local_rank(self) -> int:
        """Get local rank on current node.
 
        Returns:
            Local rank (0 if not distributed).
        """
        return self.local_rank if self._is_distributed and self.local_rank is not None else 0
 
    def barrier(self) -> None:
        """Synchronization barrier for all processes."""
        if self._is_distributed:
            import torch.distributed as dist
            dist.barrier()
 
    def all_reduce(self, tensor, op: str = 'sum') -> None:
        """All-reduce operation.
 
        Args:
            tensor: Tensor to reduce.
            op: Reduction operation ('sum', 'mean', 'max', 'min').
        """
        if not self._is_distributed:
            return
 
        import torch.distributed as dist
 
        op_map = {
            'sum': dist.ReduceOp.SUM,
            'mean': dist.ReduceOp.SUM,
            'max': dist.ReduceOp.MAX,
            'min': dist.ReduceOp.MIN
        }
 
        dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))
 
        if op == 'mean':
            tensor.div_(self.world_size)
 
    def all_gather(self, tensor: Any) -> List[Any]:
        """All-gather operation.
 
        Args:
            tensor: Tensor to gather.
 
        Returns:
            List of tensors from all processes.
        """
        if not self._is_distributed:
            return [tensor]
 
        import torch.distributed as dist
 
        tensor_list = [tensor.clone() for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
 
    def broadcast(self, tensor: Any, src: int = 0) -> None:
        """Broadcast tensor from source to all processes.
 
        Args:
            tensor: Tensor to broadcast.
            src: Source rank.
        """
        if not self._is_distributed:
            return
 
        import torch.distributed as dist
        dist.broadcast(tensor, src=src)
 
    def reduce_dict(self, data: dict, average: bool = True) -> dict:
        """Reduce dictionary values across processes.
 
        Args:
            data: Dictionary with tensor values.
            average: Whether to average values.
 
        Returns:
            Reduced dictionary.
        """
        if not self._is_distributed:
            return data
 
        import torch
 
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.all_reduce(value)
                if average:
                    value = value / self.world_size
                result[key] = value
            else:
                result[key] = value
 
        return result
 
    def destroy(self) -> None:
        """Destroy distributed process group."""
        if self._is_distributed:
            import torch.distributed as dist
            dist.destroy_process_group()
            self._is_distributed = False
            logger.info("Distributed process group destroyed")
 
    def init_from_device_manager(self, device_manager) -> None:
        """Initialize distributed environment from DeviceManager.
 
        This method integrates with DeviceManager's training mode to determine
        whether distributed initialization is needed.
 
        Args:
            device_manager: DeviceManager instance with training mode set.
 
        Note:
            - SINGLE mode: No distributed initialization needed.
            - SINGLE_MACHINE_MULTI_DEVICE mode: DDP initialization requires
              distributed launcher (torchrun or DistributedLauncher).
            - MULTI_MACHINE_MULTI_DEVICE mode: Initializes distributed process group.
 
        Important:
            For SINGLE_MACHINE_MULTI_DEVICE (multi-NPU/GPU) DDP training, you can
            use one of the following methods:
 
            1. Auto-launch (recommended): Just run `python train.py`, the framework
               will automatically detect multiple devices and launch DDP.
 
            2. Manual torchrun: `torchrun --nproc_per_node=<device_count> train.py`
 
            3. Disable auto-launch in config:
               distributed:
                 auto_launch: false
 
            The auto-launch feature uses `torch.multiprocessing.spawn` to create
            multiple processes, one for each device.
        """
        from utils.device_management import TrainingMode
 
        mode = device_manager.training_mode
 
        if mode == TrainingMode.SINGLE:
            # Single device mode - no distributed initialization needed
            logger.info("Single device mode - distributed not initialized")
            return
 
        elif mode == TrainingMode.SINGLE_MACHINE_MULTI_DEVICE:
            # DDP mode for single machine multi-device
            # Check if WORLD_SIZE is already set by launcher (torchrun or DistributedLauncher)
            existing_world_size = os.environ.get("WORLD_SIZE")
 
            if existing_world_size and int(existing_world_size) > 1:
                # Launched via distributed launcher, initialize DDP
                config = device_manager.get_device_config()
 
                if "MASTER_ADDR" not in os.environ:
                    os.environ["MASTER_ADDR"] = "localhost"
                if "MASTER_PORT" not in os.environ:
                    os.environ["MASTER_PORT"] = "29500"
 
                self.init_distributed()
 
                logger.info(
                    f"Single machine DDP mode initialized with {existing_world_size} processes"
                )
            else:
                # This should not happen if auto_launch is enabled
                # The user may have disabled auto_launch
                logger.info(
                    f"Single machine multi-device mode detected. "
                    f"For DDP training, either use 'auto_launch: true' in config "
                    f"or launch with: torchrun --nproc_per_node={len(device_manager.device_ids)} train.py"
                )
 
        elif mode == TrainingMode.MULTI_MACHINE_MULTI_DEVICE:
            # DDP mode for multi-machine
            config = device_manager.get_device_config()
 
            # Set environment variables from device config if not already set
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = config.master_addr
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(config.master_port)
 
            self.init_distributed()
 
            logger.info(
                f"DDP mode initialized - rank={self.rank}/{self.world_size}, "
                f"local_rank={self.local_rank}"
            )
 
 
# Convenience functions
def get_dist_manager() -> DistributedManager:
    """Get distributed manager instance.
 
    Returns:
        DistributedManager instance.
    """
    return DistributedManager()
 
 
def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return get_dist_manager().is_distributed()
 
 
def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_dist_manager().is_main_process()
 
 
def get_rank() -> int:
    """Get current process rank."""
    return get_dist_manager().get_rank()
 
 
def get_world_size() -> int:
    """Get total number of processes."""
    return get_dist_manager().get_world_size()
 
 
def get_local_rank() -> int:
    """Get local rank on current node."""
    return get_dist_manager().get_local_rank()
 
 
def barrier() -> None:
    """Synchronization barrier."""
    get_dist_manager().barrier()
 
 
def all_reduce(tensor, op: str = 'sum') -> None:
    """All-reduce operation."""
    get_dist_manager().all_reduce(tensor, op)
 
 
def all_gather(tensor) -> List[Any]:
    """All-gather operation."""
    return get_dist_manager().all_gather(tensor)
 
 
def broadcast(tensor, src: int = 0) -> None:
    """Broadcast operation."""
    get_dist_manager().broadcast(tensor, src)