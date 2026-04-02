"""IO utilities module.
 
Provides unified file and serialization operations.
"""
import os
import pickle
import json
import yaml
from pathlib import Path
from typing import Any, Union, Optional, Dict
import shutil
 
from .exception import MLFileNotFoundError, FileAccessError, SerializationError
 
 
PathLike = Union[str, Path]
 
 
def ensure_dir(path: PathLike) -> Path:
    """Ensure directory exists, create if not.
 
    Args:
        path: Directory path.
 
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
 
 
def copy_file(src: PathLike, dst: PathLike) -> None:
    """Copy file from source to destination.
 
    Args:
        src: Source path.
        dst: Destination path.
 
    Raises:
        MLFileNotFoundError: If source not found.
        FileAccessError: If copy fails.
    """
    src = Path(src)
    dst = Path(dst)
 
    if not src.exists():
        raise MLFileNotFoundError(f"Source file not found: {src}")
 
    try:
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
    except Exception as e:
        raise FileAccessError(f"Failed to copy {src} to {dst}: {e}")
 
 
def move_file(src: PathLike, dst: PathLike) -> None:
    """Move file from source to destination.
 
    Args:
        src: Source path.
        dst: Destination path.
 
    Raises:
        MLFileNotFoundError: If source not found.
        FileAccessError: If move fails.
    """
    src = Path(src)
    dst = Path(dst)
 
    if not src.exists():
        raise MLFileNotFoundError(f"Source file not found: {src}")
 
    try:
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
    except Exception as e:
        raise FileAccessError(f"Failed to move {src} to {dst}: {e}")
 
 
def remove_file(path: PathLike) -> None:
    """Remove file.
 
    Args:
        path: File path.
 
    Raises:
        FileAccessError: If removal fails.
    """
    path = Path(path)
 
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as e:
            raise FileAccessError(f"Failed to remove {path}: {e}")
 
 
def list_files(directory: PathLike,
               pattern: str = '*',
               recursive: bool = False) -> list:
    """List files in directory.
 
    Args:
        directory: Directory path.
        pattern: Glob pattern.
        recursive: Whether to search recursively.
 
    Returns:
        List of file paths.
    """
    directory = Path(directory)
 
    if not directory.exists():
        return []
 
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))
 
 
# Serialization functions
def save_json(data: Any, path: PathLike, indent: int = 2) -> None:
    """Save data as JSON.
 
    Args:
        data: Data to save.
        path: Output path.
        indent: JSON indentation.
 
    Raises:
        SerializationError: If serialization fails.
    """
    path = Path(path)
    ensure_dir(path.parent)
 
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise SerializationError(f"Failed to save JSON to {path}: {e}")
 
 
def load_json(path: PathLike) -> Any:
    """Load JSON file.
 
    Args:
        path: JSON file path.
 
    Returns:
        Loaded data.
 
    Raises:
        MLFileNotFoundError: If file not found.
        SerializationError: If deserialization fails.
    """
    path = Path(path)
 
    if not path.exists():
        raise MLFileNotFoundError(f"JSON file not found: {path}")
 
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load JSON from {path}: {e}")
 
 
def save_yaml(data: Any, path: PathLike) -> None:
    """Save data as YAML.
 
    Args:
        data: Data to save.
        path: Output path.
 
    Raises:
        SerializationError: If serialization fails.
    """
    path = Path(path)
    ensure_dir(path.parent)
 
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise SerializationError(f"Failed to save YAML to {path}: {e}")
 
 
def load_yaml(path: PathLike) -> Any:
    """Load YAML file.
 
    Args:
        path: YAML file path.
 
    Returns:
        Loaded data.
 
    Raises:
        MLFileNotFoundError: If file not found.
        SerializationError: If deserialization fails.
    """
    path = Path(path)
 
    if not path.exists():
        raise MLFileNotFoundError(f"YAML file not found: {path}")
 
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load YAML from {path}: {e}")
 
 
def save_pickle(data: Any, path: PathLike) -> None:
    """Save data using pickle.
 
    Args:
        data: Data to save.
        path: Output path.
 
    Raises:
        SerializationError: If serialization fails.
    """
    path = Path(path)
    ensure_dir(path.parent)
 
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise SerializationError(f"Failed to save pickle to {path}: {e}")
 
 
def load_pickle(path: PathLike) -> Any:
    """Load pickle file.
 
    Args:
        path: Pickle file path.
 
    Returns:
        Loaded data.
 
    Raises:
        MLFileNotFoundError: If file not found.
        SerializationError: If deserialization fails.
    """
    path = Path(path)
 
    if not path.exists():
        raise MLFileNotFoundError(f"Pickle file not found: {path}")
 
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load pickle from {path}: {e}")
 
 
# Checkpoint utilities
def _extract_epoch_from_path(path: Path) -> int:
    """Extract epoch number from checkpoint path.
 
    Args:
        path: Checkpoint file path.
 
    Returns:
        Epoch number, or -1 if extraction fails.
    """
    # File name format: checkpoint_epoch_{epoch}.pth
    name = path.stem  # checkpoint_epoch_0
    try:
        return int(name.split('_')[-1])
    except (ValueError, IndexError):
        return -1
 
 
class CheckpointManager:
    """Manager for model checkpoints."""
 
    def __init__(self, checkpoint_dir: PathLike, max_keep: int = 5):
        """Initialize checkpoint manager.
 
        Args:
            checkpoint_dir: Directory to save checkpoints.
            max_keep: Maximum checkpoints to keep.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        ensure_dir(self.checkpoint_dir)
 
    def save(self,
             state_dict: Dict[str, Any],
             epoch: int,
             is_best: bool = False) -> Path:
        """Save checkpoint.
 
        Args:
            state_dict: State dictionary to save.
            epoch: Current epoch.
            is_best: Whether this is the best checkpoint.
 
        Returns:
            Path to saved checkpoint.
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
 
        save_pickle(state_dict, checkpoint_path)
 
        # Save as best if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            copy_file(checkpoint_path, best_path)
 
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
 
        return checkpoint_path
 
    def load(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Load checkpoint.
 
        Args:
            epoch: Specific epoch to load. If None, loads latest.
 
        Returns:
            Checkpoint state dictionary.
        """
        if epoch is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        else:
            # Find latest checkpoint by epoch number
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
                key=lambda p: _extract_epoch_from_path(p)
            )
            if not checkpoints:
                raise MLFileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]  # Latest by epoch number
 
        return load_pickle(checkpoint_path)
 
    def load_best(self) -> Dict[str, Any]:
        """Load best checkpoint.
 
        Returns:
            Best checkpoint state dictionary.
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if not best_path.exists():
            raise MLFileNotFoundError("Best checkpoint not found")
        return load_pickle(best_path)
 
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only max_keep."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: _extract_epoch_from_path(p)
        )
 
        while len(checkpoints) > self.max_keep:
            old_checkpoint = checkpoints.pop(0)
            remove_file(old_checkpoint)
 
    def list_checkpoints(self) -> list:
        """List all available checkpoints.
 
        Returns:
            List of checkpoint paths.
        """
        return list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))