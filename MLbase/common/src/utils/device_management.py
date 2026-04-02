"""Device management module.
 
Automatically detects and manages compute devices.
Priority: NPU -> GPU -> CPU
 
Supports three training modes:
- SINGLE: Single NPU, GPU or CPU training
- SINGLE_MACHINE_MULTI_DEVICE: Multiple NPU/GPU devices on one machine using DDP
- MULTI_MACHINE_MULTI_DEVICE: Multiple machines with multiple NPU/GPU devices using DDP
"""
import os
from typing import List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
 
 
class DeviceType(Enum):
    """Device types."""
    NPU = "npu"
    CUDA = "cuda"
    CPU = "cpu"
 
 
class TrainingMode(Enum):
    """Training mode enumeration.
 
    Defines the supported training modes:
    - AUTO: Automatically select optimal mode based on available devices
    - SINGLE: Single device (NPU, GPU or CPU) training
    - SINGLE_MACHINE_MULTI_DEVICE: Multiple devices on one machine using DDP
    - MULTI_MACHINE_MULTI_DEVICE: Multiple machines with multiple devices using DDP
    """
    AUTO = "auto"
    SINGLE = "single"
    SINGLE_MACHINE_MULTI_DEVICE = "single_multi"
    MULTI_MACHINE_MULTI_DEVICE = "multi_multi"
 
 
@dataclass
class DeviceConfig:
    """Device configuration for training.
 
    Attributes:
        mode: Training mode (single, single_multi, multi_multi).
        device_ids: List of device IDs to use.
        local_rank: Local rank for distributed training.
        master_addr: Master node address for distributed training.
        master_port: Master node port for distributed training.
    """
    mode: TrainingMode = TrainingMode.SINGLE
    device_ids: List[int] = field(default_factory=list)
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
 
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.device_ids:
            if self.mode == TrainingMode.SINGLE:
                self.device_ids = [0]
            # For multi-GPU modes, device_ids should be set explicitly
            # or auto-detected later
 
 
class Device:
    """Device representation."""
 
    def __init__(self, device_type: DeviceType, index: int = 0):
        """Initialize device.
 
        Args:
            device_type: Type of device.
            index: Device index for multi-device types.
        """
        self.type = device_type
        self.index = index
        self._name = None
 
    @property
    def name(self) -> str:
        """Get device name string."""
        if self._name is None:
            if self.type == DeviceType.CPU:
                self._name = "cpu"
            else:
                self._name = f"{self.type.value}:{self.index}"
        return self._name
 
    def __str__(self) -> str:
        """String representation."""
        return self.name
 
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Device({self.name})"
 
    def __eq__(self, other) -> bool:
        """Equality check."""
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return self.name == other
        return False
 
 
class DeviceManager:
    """Manages device detection and selection.
 
    Singleton pattern ensures consistent device management across the framework.
    Supports three training modes: single GPU, single machine multi-GPU, and
    multi-machine multi-GPU.
    """
 
    _instance: Optional['DeviceManager'] = None
 
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
 
    def __init__(self, preferred_device: Optional[str] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
 
        self._preferred_device = preferred_device
        self._current_device: Optional[Device] = None
        self._available_devices: List[Device] = []
 
        self._training_mode: TrainingMode = TrainingMode.SINGLE
        self._device_ids: List[int] = []
        self._local_rank: int = 0
 
        self._detect_devices()
        self._initialized = True
 
    @classmethod
    def reset(cls) -> None:
        """Reset singleton state for testing or re-initialization."""
        cls._instance = None
 
    def _detect_devices(self) -> None:
        """Detect available devices."""
        self._available_devices = []
 
        # Check for NPU (Ascend) using torch_npu
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                npu_count = torch_npu.npu.device_count()
                for i in range(npu_count):
                    self._available_devices.append(Device(DeviceType.NPU, i))
        except ImportError:
            pass
 
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    self._available_devices.append(Device(DeviceType.CUDA, i))
        except ImportError:
            pass
 
        # CPU is always available
        self._available_devices.append(Device(DeviceType.CPU, 0))
 
        # Set current device based on preference or priority
        self._set_current_device()
 
    def _set_current_device(self) -> None:
        """Set current device based on preference or priority."""
        if self._preferred_device:
            device_type = self._preferred_device.lower()
 
            if device_type == 'npu':
                devices = self.get_available_devices('npu')
                if devices:
                    self._current_device = devices[0]
                    return
                raise RuntimeError("NPU requested but not available")
 
            elif device_type in ('cuda', 'gpu'):
                devices = self.get_available_devices('cuda')
                if devices:
                    self._current_device = devices[0]
                    return
                raise RuntimeError("CUDA requested but not available")
 
            elif device_type == 'cpu':
                devices = self.get_available_devices('cpu')
                if devices:
                    self._current_device = devices[0]
                return
 
        # Auto-select using priority
        _, devices = self._get_priority_accelerator()
        self._current_device = devices[0]
 
    def get_current_device(self) -> Device:
        """Get current device.
 
        Returns:
            Current device.
        """
        return self._current_device
 
    def get_available_devices(self, device_type: Optional[str] = None) -> List[Device]:
        """Get available devices.
 
        Args:
            device_type: Optional filter by device type.
 
        Returns:
            List of available devices.
        """
        if device_type:
            device_type = device_type.lower()
            return [d for d in self._available_devices if d.type.value == device_type]
        return self._available_devices.copy()
 
    def set_device(self, device: Union[str, Device]) -> None:
        """Set current device.
 
        Args:
            device: Device string (e.g., 'cuda:0') or Device object.
 
        Raises:
            ValueError: If device not available.
        """
        if isinstance(device, str):
            # Parse device string
            if ':' in device:
                device_type, index = device.split(':')
                index = int(index)
            else:
                device_type = device
                index = 0
 
            device_type_enum = DeviceType(device_type.lower())
            device = Device(device_type_enum, index)
 
        # Verify device is available
        if device not in self._available_devices:
            raise ValueError(f"Device {device} not available. "
                           f"Available: {[str(d) for d in self._available_devices]}")
 
        self._current_device = device
 
        # Set environment for frameworks
        if device.type == DeviceType.NPU:
            os.environ["ASCEND_VISIBLE_DEVICES"] = str(device.index)
        elif device.type == DeviceType.CUDA:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device.index)
 
    def get_device_count(self, device_type: Optional[str] = None) -> int:
        """Get count of available devices.
 
        Args:
            device_type: Optional device type filter.
 
        Returns:
            Number of available devices.
        """
        return len(self.get_available_devices(device_type))
 
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return any(d.type == DeviceType.CUDA for d in self._available_devices)
 
    @property
    def is_npu_available(self) -> bool:
        """Check if NPU is available."""
        return any(d.type == DeviceType.NPU for d in self._available_devices)
 
    # ==================== New Training Mode Support ====================
 
    @property
    def training_mode(self) -> TrainingMode:
        """Get current training mode."""
        return self._training_mode
 
    @property
    def device_ids(self) -> List[int]:
        """Get list of device IDs for current training mode."""
        return self._device_ids.copy()
 
    @property
    def local_rank(self) -> int:
        """Get local rank for distributed training."""
        return self._local_rank
 
    def set_training_mode(self,
                          mode: TrainingMode,
                          device_ids: Optional[List[int]] = None) -> None:
        """Set training mode and configure devices.
 
        Args:
            mode: Training mode to set.
            device_ids: Optional list of device IDs. If None, auto-detected.
 
        Raises:
            ValueError: If specified devices are not available.
        """
        if mode == TrainingMode.AUTO:
            mode = self._auto_select_training_mode()
 
        self._training_mode = mode
 
        handlers = {
            TrainingMode.SINGLE: self._setup_single_mode,
            TrainingMode.SINGLE_MACHINE_MULTI_DEVICE: self._setup_single_machine_multi_mode,
            TrainingMode.MULTI_MACHINE_MULTI_DEVICE: self._setup_multi_machine_multi_mode,
        }
        handlers[mode](device_ids)
        self._update_accelerator_visible_devices()
 
    def _setup_single_mode(self, device_ids: Optional[List[int]]) -> None:
        """Setup for single device training."""
        self._device_ids = [self._select_best_device_index()]
        self._current_device = Device(
            self._get_accelerator_device_type(),
            self._device_ids[0]
        )
 
    def _setup_single_machine_multi_mode(self, device_ids: Optional[List[int]]) -> None:
        """Setup for single machine multi-device training."""
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
 
        if device_ids is not None:
            self._validate_accelerator_ids(device_ids)
            self._device_ids = device_ids
        else:
            self._device_ids = self._get_all_accelerator_ids()
 
        if not self._device_ids:
            raise ValueError("No accelerator devices (NPU/GPU) available for multi-device training")
 
        if self._local_rank >= len(self._device_ids):
            raise ValueError(
                f"LOCAL_RANK {self._local_rank} exceeds available accelerator devices {len(self._device_ids)}"
            )
 
        self._current_device = Device(
            self._get_accelerator_device_type(),
            self._device_ids[self._local_rank]
        )
 
    def _setup_multi_machine_multi_mode(self, device_ids: Optional[List[int]]) -> None:
        """Setup for multi-machine multi-device training."""
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._device_ids = [self._local_rank]
 
        device_type = self._get_accelerator_device_type()
        device_count = self.get_device_count(device_type.value)
 
        if self._local_rank >= device_count:
            raise ValueError(
                f"LOCAL_RANK {self._local_rank} exceeds available {device_type.value} devices"
            )
 
        self._current_device = Device(device_type, self._local_rank)
 
    def _auto_select_training_mode(self) -> TrainingMode:
        """Auto select best training mode for maximum efficiency.
 
        Decision logic:
        1. Check WORLD_SIZE env var for distributed setup
        2. If WORLD_SIZE > 1, check MASTER_ADDR to distinguish single vs multi machine
        3. If WORLD_SIZE == 1, check available accelerator count
 
        Returns:
            Optimal TrainingMode.
        """
        world_size = int(os.environ.get("WORLD_SIZE", 1))
 
        if world_size > 1:
            # Distinguish single-machine multi-device vs multi-machine multi-device
            # by checking if MASTER_ADDR points to localhost
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            if master_addr in ("localhost", "127.0.0.1"):
                return TrainingMode.SINGLE_MACHINE_MULTI_DEVICE
            else:
                return TrainingMode.MULTI_MACHINE_MULTI_DEVICE
 
        accelerator_count = len(self._get_all_accelerator_ids())
 
        if accelerator_count > 1:
            return TrainingMode.SINGLE_MACHINE_MULTI_DEVICE
        else:
            return TrainingMode.SINGLE
 
    def setup_for_distributed(self) -> None:
        """Setup device for distributed training.
 
        Automatically detects training mode and configures devices accordingly.
        This method should be called before initializing distributed training.
        Uses AUTO mode by default for maximum training efficiency.
 
        For distributed modes (SINGLE_MACHINE_MULTI_DEVICE and MULTI_MACHINE_MULTI_DEVICE),
        each process uses the device corresponding to its LOCAL_RANK.
        """
        # Use AUTO mode to automatically select optimal mode
        # set_training_mode() will handle LOCAL_RANK-based device selection
        self.set_training_mode(TrainingMode.AUTO)
 
    def get_device_config(self) -> DeviceConfig:
        """Get current device configuration.
 
        Returns:
            DeviceConfig with current settings.
        """
        return DeviceConfig(
            mode=self._training_mode,
            device_ids=self._device_ids.copy(),
            local_rank=self._local_rank,
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=int(os.environ.get("MASTER_PORT", "29500"))
        )
 
    # ==================== Helper Methods ====================
 
    def _get_priority_accelerator(self) -> Tuple[DeviceType, List[Device]]:
        """Get highest priority accelerator (NPU > CUDA > CPU) and its devices.
 
        Returns:
            Tuple of (DeviceType, List[Device]).
        """
        if self.is_npu_available:
            return DeviceType.NPU, self.get_available_devices('npu')
        if self.is_cuda_available:
            return DeviceType.CUDA, self.get_available_devices('cuda')
        return DeviceType.CPU, [Device(DeviceType.CPU, 0)]
 
    def _select_best_device_index(self) -> int:
        """Select the best available device index.
 
        Returns:
            Device index of the best available device (0 for CPU).
        """
        _, devices = self._get_priority_accelerator()
        return devices[0].index if devices else 0
 
    def _get_all_accelerator_ids(self) -> List[int]:
        """Get list of all available accelerator device IDs (NPU first, then GPU).
 
        Returns:
            List of accelerator device indices.
        """
        device_type, devices = self._get_priority_accelerator()
        return [d.index for d in devices] if device_type != DeviceType.CPU else []
 
    def _get_accelerator_device_type(self) -> DeviceType:
        """Get accelerator device type based on priority (NPU -> CUDA).
 
        Returns:
            DeviceType of available accelerator, or CPU if none.
        """
        device_type, _ = self._get_priority_accelerator()
        return device_type
 
    def _validate_accelerator_ids(self, device_ids: List[int]) -> None:
        """Validate that all specified accelerator device IDs are available.
 
        Args:
            device_ids: List of device IDs to validate.
 
        Raises:
            ValueError: If any device ID is not available.
        """
        device_type = self._get_accelerator_device_type()
        available_ids = set(d.index for d in self.get_available_devices(device_type.value))
        for device_id in device_ids:
            if device_id not in available_ids:
                raise ValueError(
                    f"{device_type.value} device {device_id} not available. "
                    f"Available: {sorted(available_ids)}"
                )
 
    def _update_accelerator_visible_devices(self) -> None:
        """Update accelerator visible devices environment variable based on device_ids.
 
        Sets appropriate environment variable based on device type:
        - NPU: ASCEND_VISIBLE_DEVICES
        - CUDA: CUDA_VISIBLE_DEVICES
        """
        if not self._device_ids:
            return
 
        device_type = self._get_accelerator_device_type()
        device_ids_str = ",".join(map(str, self._device_ids))
 
        if device_type == DeviceType.NPU:
            os.environ["ASCEND_VISIBLE_DEVICES"] = device_ids_str
        elif device_type == DeviceType.CUDA:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
 
 
def get_device_manager(preferred_device: Optional[str] = None) -> DeviceManager:
    """Get device manager instance.
 
    Args:
        preferred_device: Optional preferred device type.
 
    Returns:
        DeviceManager singleton instance.
    """
    return DeviceManager(preferred_device)
 
 
def get_device(preferred_device: Optional[str] = None) -> Device:
    """Get current device.
 
    Convenience function to get current device.
 
    Args:
        preferred_device: Optional preferred device.
 
    Returns:
        Current device.
    """
    return get_device_manager(preferred_device).get_current_device()
 
 
def get_training_mode() -> TrainingMode:
    """Get current training mode.
 
    Returns:
        Current TrainingMode.
    """
    return get_device_manager().training_mode
 
 
def get_device_ids() -> List[int]:
    """Get device IDs for current training mode.
 
    Returns:
        List of device IDs.
    """
    return get_device_manager().device_ids