"""Distributed training launcher module.
 
Provides automatic multi-process launching for DDP training on single machine
with multiple devices (NPU/GPU).
 
This module enables users to run `python train.py` directly without manually
using `torchrun`. The system will automatically detect multiple devices and
launch the appropriate number of processes.
"""
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Callable, Any, Optional, Dict, Tuple
 
from utils.logger import get_logger
 
 
logger = get_logger()
 
 
class DistributedLauncher:
    """Automatic distributed training launcher.
 
    Automatically detects multi-device environments and launches multiple
    processes for DDP training when needed.
 
    Usage:
        launcher = DistributedLauncher(config)
        if launcher.should_launch():
            launcher.launch(train_fn, train_config)
        else:
            train_fn(train_config)  # Single process mode
 
    Environment Variables:
        ML_FRAMEWORK_LAUNCHED: Set when launched by this framework
        WORLD_SIZE: Number of processes (set by torchrun or this launcher)
        RANK: Global rank of current process
        LOCAL_RANK: Local rank on current node
        MASTER_ADDR: Master node address
        MASTER_PORT: Master node port
    """
 
    ENV_LAUNCHED = 'ML_FRAMEWORK_LAUNCHED'
    ENV_WORLD_SIZE = 'WORLD_SIZE'
    ENV_RANK = 'RANK'
    ENV_LOCAL_RANK = 'LOCAL_RANK'
    ENV_MASTER_ADDR = 'MASTER_ADDR'
    ENV_MASTER_PORT = 'MASTER_PORT'
 
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize launcher.
 
        Args:
            config: Launcher configuration containing:
                - master_port: Port for distributed communication (default: 29500)
                - master_addr: Address for distributed communication (default: localhost)
                - auto_launch: Whether to auto-launch (default: True)
        """
        self.config = config or {}
        self.master_port = self.config.get('master_port', 29500)
        self.master_addr = self.config.get('master_addr', 'localhost')
        self._launched = False
        self._result_file: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None
 
    def is_launched_by_framework(self) -> bool:
        """Check if current process was launched by this framework.
 
        Returns:
            True if launched by DistributedLauncher.
        """
        return os.environ.get(self.ENV_LAUNCHED) == '1'
 
    def is_launched_by_torchrun(self) -> bool:
        """Check if current process was launched by torchrun.
 
        Returns:
            True if launched by torchrun or torch.distributed.launch.
        """
        return os.environ.get(self.ENV_WORLD_SIZE) is not None
 
    def get_device_count(self) -> int:
        """Get number of available accelerator devices.
 
        Returns:
            Number of NPU or GPU devices available.
        """
        from utils.device_management import get_device_manager
        dm = get_device_manager()
        npu_count = dm.get_device_count('npu')
        if npu_count > 0:
            return npu_count
        return dm.get_device_count('cuda')
 
    def should_launch(self) -> bool:
        """Check if multi-process launch is needed.
 
        Returns:
            True if should launch multiple processes for DDP.
        """
        # Already launched by framework or torchrun
        if self.is_launched_by_framework() or self.is_launched_by_torchrun():
            return False
 
        # Check if auto_launch is enabled
        auto_launch = self.config.get('auto_launch', True)
        if not auto_launch:
            return False
 
        # Check device count
        device_count = self.get_device_count()
        return device_count > 1
 
    def launch(self, main_fn: Callable, config: Any) -> bool:
        """Launch multi-process training.
 
        Args:
            main_fn: Main training function to run in each process.
            config: Configuration to pass to main_fn.
 
        Returns:
            True if multi-process was launched, False otherwise.
        """
        if not self.should_launch():
            return False
 
        device_count = self.get_device_count()
 
        logger.info(
            f"Auto-launching {device_count} processes for distributed training"
        )
 
        # Create temporary file for collecting results from rank 0
        self._result_file = os.path.join(
            tempfile.gettempdir(),
            f'dist_result_{os.getpid()}_{id(self)}.json'
        )
        os.environ['ML_FRAMEWORK_RESULT_FILE'] = self._result_file
 
        # Set environment variables for distributed training
        os.environ[self.ENV_MASTER_ADDR] = self.master_addr
        os.environ[self.ENV_MASTER_PORT] = str(self.master_port)
        os.environ[self.ENV_WORLD_SIZE] = str(device_count)
        os.environ[self.ENV_LAUNCHED] = '1'
 
        try:
            import torch.multiprocessing as mp
 
            # Launch processes
            mp.spawn(
                self._worker_entry,
                args=(main_fn, config),
                nprocs=device_count,
                join=True
            )
            self._launched = True
 
            # Collect results from rank 0
            self._collect_results()
 
            return True
 
        except ImportError:
            logger.warning(
                "torch.multiprocessing not available, falling back to single process"
            )
            return False
        finally:
            # Cleanup result file
            if self._result_file and os.path.exists(self._result_file):
                try:
                    os.remove(self._result_file)
                except Exception:
                    pass
 
    def _collect_results(self) -> None:
        """Collect training results from rank 0 process."""
        if not self._result_file or not os.path.exists(self._result_file):
            logger.warning("No result file found from distributed training")
            self._result = None
            return
 
        try:
            with open(self._result_file, 'r', encoding='utf-8') as f:
                self._result = json.load(f)
            logger.info("Training results collected from distributed processes")
        except Exception as e:
            logger.warning(f"Failed to collect training results: {e}")
            self._result = None
 
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get training result collected from rank 0 process.
 
        Returns:
            Training history dictionary or None if not available.
        """
        return self._result
 
    def _worker_entry(self, rank: int, main_fn: Callable, config: Any) -> None:
        """Entry point for each worker process.
 
        Args:
            rank: Rank of this worker process.
            main_fn: Main training function.
            config: Configuration to pass to main_fn.
        """
        # Reset singleton states for this process
        # This is necessary because mp.spawn may inherit parent process state
        from utils.device_management import DeviceManager
        from utils.distributed_comm import DistributedManager
 
        DeviceManager.reset()
        DistributedManager.reset()
 
        # Set rank environment variables
        os.environ[self.ENV_RANK] = str(rank)
        os.environ[self.ENV_LOCAL_RANK] = str(rank)
 
        logger.info(f"Worker {rank} started")
 
        try:
            result = main_fn(config)
 
            # Save result from rank 0 to file for parent process to collect
            if rank == 0 and result is not None:
                result_file = os.environ.get('ML_FRAMEWORK_RESULT_FILE')
                if result_file:
                    try:
                        # Convert result to JSON-serializable format
                        serializable_result = self._make_serializable(result)
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(serializable_result, f)
                        logger.info("Rank 0: Training result saved")
                    except Exception as e:
                        logger.warning(f"Rank 0: Failed to save training result: {e}")
 
        except Exception as e:
            logger.error(f"Worker {rank} failed: {e}")
            raise
 
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.
 
        Args:
            obj: Object to convert.
 
        Returns:
            JSON-serializable version of the object.
        """
        import numpy as np
 
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, 'item'):  # PyTorch tensor scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # PyTorch tensor
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except Exception:
                return None
 
    def get_launch_info(self) -> Dict[str, Any]:
        """Get current launch information.
 
        Returns:
            Dictionary with launch information.
        """
        return {
            'launched_by_framework': self.is_launched_by_framework(),
            'launched_by_torchrun': self.is_launched_by_torchrun(),
            'world_size': int(os.environ.get(self.ENV_WORLD_SIZE, 1)),
            'rank': int(os.environ.get(self.ENV_RANK, 0)),
            'local_rank': int(os.environ.get(self.ENV_LOCAL_RANK, 0)),
            'master_addr': os.environ.get(self.ENV_MASTER_ADDR, 'localhost'),
            'master_port': int(os.environ.get(self.ENV_MASTER_PORT, 29500)),
        }
 
 
def launch_distributed_if_needed(
    main_fn: Callable,
    config: Any,
    dist_config: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to launch distributed training if needed.
 
    Args:
        main_fn: Main training function.
        config: Configuration to pass to main_fn.
        dist_config: Distributed configuration.
 
    Returns:
        True if multi-process was launched, False if running in single process.
    """
    launcher = DistributedLauncher(dist_config)
 
    if launcher.should_launch():
        return launcher.launch(main_fn, config)
 
    return False
 
 
def is_distributed_launched() -> bool:
    """Check if distributed training was launched.
 
    Returns:
        True if running in distributed mode.
    """
    launcher = DistributedLauncher()
    return launcher.is_launched_by_framework() or launcher.is_launched_by_torchrun()
 
 
def get_rank() -> int:
    """Get current process rank.
 
    Returns:
        Rank of current process (0 if not distributed).
    """
    return int(os.environ.get(DistributedLauncher.ENV_RANK, 0))
 
 
def get_local_rank() -> int:
    """Get local rank on current node.
 
    Returns:
        Local rank (0 if not distributed).
    """
    return int(os.environ.get(DistributedLauncher.ENV_LOCAL_RANK, 0))
 
 
def get_world_size() -> int:
    """Get total number of processes.
 
    Returns:
        World size (1 if not distributed).
    """
    return int(os.environ.get(DistributedLauncher.ENV_WORLD_SIZE, 1))