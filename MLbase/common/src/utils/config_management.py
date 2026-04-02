"""Configuration management module.
 
Supports YAML/JSON configs with environment variable overrides.
Also supports `_target_` for dynamic object instantiation.
"""
import os
import yaml
import json
import importlib
import inspect
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
 
 
def filter_init_params(cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters that the class constructor accepts.
 
    Args:
        cls: Target class.
        params: Parameter dictionary to filter.
 
    Returns:
        Filtered parameter dictionary containing only valid constructor params.
    """
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    return {
        k: v for k, v in params.items()
        if k in valid_params or 'kwargs' in valid_params
    }
 
 
def instantiate(config: Dict[str, Any], **kwargs) -> Any:
    """Instantiate an object from config with `_target_` field.
 
    This function allows dynamic instantiation of any class from configuration.
    It recursively instantiates nested configs that have `_target_` fields.
 
    Use this for dynamic instantiation when the class path is known at
    configuration time. Supports any importable class without requiring
    prior registration in a registry.
 
    Note:
        For registered framework components (models, optimizers, losses, etc.),
        prefer Registry.build() which validates against the registry and provides
        better error messages for unknown types.
 
    Args:
        config: Configuration dict with `_target_` field and other parameters.
               Format: {'_target_': 'module.path.ClassName', 'arg1': val1, ...}
        **kwargs: Additional keyword arguments to pass to the constructor.
 
    Returns:
        Instantiated object.
 
    Raises:
        ValueError: If `_target_` not in config.
        ImportError: If module or class not found.
 
    Example:
        >>> config = {
        ...     '_target_': 'my_module.MyModel',
        ...     'hidden_dim': 256,
        ...     'num_layers': 3
        ... }
        >>> model = instantiate(config)
    """
    if not isinstance(config, dict):
        return config
 
    if '_target_' not in config:
        # Recursively process nested dicts
        return {k: instantiate(v) if isinstance(v, dict) else v
                for k, v in config.items()}
 
    # Get target class path
    target = config['_target_']
 
    # Parse module and class name
    if '.' not in target:
        raise ValueError(f"Invalid _target_ format: {target}. "
                        f"Expected 'module.path.ClassName'")
 
    parts = target.rsplit('.', 1)
    module_path = parts[0]
    class_name = parts[1]
 
    # Import module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
 
    # Get class
    if not hasattr(module, class_name):
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")
 
    cls = getattr(module, class_name)
 
    # Prepare parameters
    params = {k: v for k, v in config.items() if k != '_target_'}
 
    # Recursively instantiate nested configs with _target_
    for key, value in params.items():
        if isinstance(value, dict) and '_target_' in value:
            params[key] = instantiate(value)
        elif isinstance(value, list):
            params[key] = [
                instantiate(item) if isinstance(item, dict) and '_target_' in item else item
                for item in value
            ]
 
    # Merge with kwargs (kwargs take precedence)
    params.update(kwargs)
 
    # Filter parameters that the constructor accepts
    filtered_params = filter_init_params(cls, params)
 
    return cls(**filtered_params)
 
 
class Config:
    """Configuration manager with hierarchical loading.
 
    Loading priority (high to low):
    1. Environment variables
    2. Configuration file (YAML/JSON)
    3. Default values
    """
 
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
 
        Args:
            config_dict: Initial configuration dictionary.
        """
        self._config = config_dict or {}
        self._default_config = {}
 
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Config':
        """Load configuration from file.
 
        Args:
            filepath: Path to YAML or JSON file.
 
        Returns:
            Config instance.
 
        Raises:
            ValueError: If file format not supported.
            FileNotFoundError: If file doesn't exist.
        """
        filepath = Path(filepath)
 
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
 
        suffix = filepath.suffix.lower()
 
        if suffix in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
 
        return cls(config_dict or {})
 
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary.
 
        Args:
            config_dict: Configuration dictionary.
 
        Returns:
            Config instance.
        """
        return cls(config_dict)
 
    def set_default(self, default_dict: Dict[str, Any]) -> 'Config':
        """Set default configuration values.
 
        Args:
            default_dict: Default configuration dictionary.
 
        Returns:
            Self for chaining.
        """
        self._default_config = default_dict
        return self
 
    def apply_env_overrides(self, prefix: str = 'ML_') -> 'Config':
        """Apply environment variable overrides.
 
        Args:
            prefix: Environment variable prefix.
 
        Returns:
            Self for chaining.
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested keys
                # e.g., ML_TRAINING_BATCH_SIZE -> training.batch_size
                config_key = key[len(prefix):].lower().replace('_', '.')
                self._set_nested_value(config_key, self._parse_value(value))
        return self
 
    def _set_nested_value(self, key: str, value: Any) -> None:
        """Set nested dictionary value using dot notation.
 
        Args:
            key: Dot-separated key path.
            value: Value to set.
        """
        keys = key.split('.')
        current = self._config
 
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
 
        current[keys[-1]] = value
 
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type.
 
        Args:
            value: String value from environment.
 
        Returns:
            Parsed value (int, float, bool, or string).
        """
        # Try bool
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
 
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
 
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
 
        # Return as string
        return value
 
    # Sentinel value to distinguish "key not found" from "key explicitly set to None"
    _MISSING = object()
 
    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value.
 
        Args:
            key: Dot-separated key path. If None, returns entire config.
            default: Default value if key not found.
 
        Returns:
            Configuration value.
        """
        if key is None:
            # Merge default and current config
            merged = self._deep_merge(self._default_config.copy(), self._config)
            return merged
 
        # Check current config first using sentinel to detect "not found" vs "set to None"
        value = self._get_nested_value(self._config, key, self._MISSING)
        if value is not self._MISSING:
            return value
 
        # Fall back to default config or provided default
        return self._get_nested_value(self._default_config, key, default)
 
    def _get_nested_value(self, config: Dict, key: str, default: Any = None) -> Any:
        """Get nested value using dot notation.
 
        Args:
            config: Configuration dictionary.
            key: Dot-separated key path.
            default: Default value if not found.
 
        Returns:
            Value or default.
        """
        keys = key.split('.')
        current = config
 
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
 
        return current
 
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries.
 
        Args:
            base: Base dictionary.
            override: Override dictionary.
 
        Returns:
            Merged dictionary.
        """
        result = base.copy()
 
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
 
        return result
 
    def set(self, key: str, value: Any) -> 'Config':
        """Set configuration value.
 
        Args:
            key: Dot-separated key path.
            value: Value to set.
 
        Returns:
            Self for chaining.
        """
        self._set_nested_value(key, value)
        return self
 
    def merge(self, other: Union[Dict, 'Config']) -> 'Config':
        """Merge another config into this one.
 
        Args:
            other: Other Config or dict to merge.
 
        Returns:
            Self for chaining.
        """
        if isinstance(other, Config):
            other = other._config
 
        self._config = self._deep_merge(self._config, other)
        return self
 
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
 
        Returns:
            Configuration dictionary.
        """
        return self.get()
 
    def __getitem__(self, key: str) -> Any:
        """Get value using bracket notation."""
        return self.get(key)
 
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value using bracket notation."""
        self.set(key, value)
 
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_nested_value(self._config, key) is not None or \
               self._get_nested_value(self._default_config, key) is not None
 
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_dict()})"
 
 
def load_config(filepath: Optional[Union[str, Path]] = None,
                default_config: Optional[Dict] = None,
                env_prefix: str = 'ML_') -> Config:
    """Convenience function to load configuration.
 
    Args:
        filepath: Optional config file path.
        default_config: Optional default configuration.
        env_prefix: Environment variable prefix.
 
    Returns:
        Config instance.
    """
    if filepath:
        config = Config.from_file(filepath)
    else:
        config = Config()
 
    if default_config:
        config.set_default(default_config)
 
    config.apply_env_overrides(env_prefix)
 
    return config