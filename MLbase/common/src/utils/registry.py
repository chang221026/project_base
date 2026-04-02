"""Registry management module.
 
Provides a universal registration mechanism for extending framework components.
"""
from typing import Any, Callable, Dict, Optional, Type, TypeVar
import importlib
 
from utils.config_management import filter_init_params
 
 
T = TypeVar('T')
 
 
class Registry:
    """Universal registry for managing extensible components.
 
    Naming convention: "module-submodule-name-version(optional)"
    Example: "model-cnn-resnet50-v1", "optimizer-sgd-momentum"
    """
 
    def __init__(self, name: str):
        """Initialize registry.
 
        Args:
            name: Registry name identifying the component type.
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
 
    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class.
 
        Args:
            name: Optional registration name. If None, uses class name.
 
        Returns:
            Decorator function.
        """
        def decorator(cls: Type) -> Type:
            registry_name = name or cls.__name__
            if registry_name in self._registry:
                raise ValueError(
                    f"'{registry_name}' already registered in {self.name}. "
                    f"Existing: {self._registry[registry_name]}, New: {cls}"
                )
            self._registry[registry_name] = cls
            return cls
        return decorator
 
    def get(self, name: str) -> Optional[Type]:
        """Get registered class by name.
 
        Args:
            name: Registered component name.
 
        Returns:
            Registered class or None if not found.
        """
        return self._registry.get(name)
 
    def build(self, config: Dict[str, Any]) -> Any:
        """Build instance from configuration.
 
        Use this for instantiating registered framework components. The component
        must be registered in this registry before it can be built.
 
        Unlike instantiate() which uses `_target_` for dynamic imports, this
        method uses `type` to lookup components in the registry. This provides:
        - Validation that the component exists
        - Better error messages for unknown types
        - Centralized component management
 
        Args:
            config: Configuration dict containing 'type' and other parameters.
                   Format: {'type': 'registered_name', 'arg1': val1, ...}
 
        Returns:
            Instantiated component.
 
        Raises:
            KeyError: If 'type' not in config.
            ValueError: If type not registered.
        """
        if 'type' not in config:
            raise KeyError(f"Config must contain 'type' field. Got: {config}")
 
        obj_type = config['type']
        cls = self.get(obj_type)
        if cls is None:
            raise ValueError(
                f"'{obj_type}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
 
        # Remove 'type' from config and pass rest to constructor
        params = {k: v for k, v in config.items() if k != 'type'}
 
        # Filter parameters that the constructor accepts
        filtered_params = filter_init_params(cls, params)
 
        return cls(**filtered_params)
 
    def list_registered(self) -> list:
        """List all registered component names.
 
        Returns:
            List of registered names.
        """
        return list(self._registry.keys())
 
    def __contains__(self, name: str) -> bool:
        """Check if name is registered.
 
        Args:
            name: Component name to check.
 
        Returns:
            True if registered, False otherwise.
        """
        return name in self._registry
 
    def __repr__(self) -> str:
        """String representation of registry."""
        return f"Registry({self.name}): {self.list_registered()}"
 
 
def build_from_cfg(config: Dict[str, Any], registry: Registry) -> Any:
    """Build object from config using registry.
 
    Args:
        config: Configuration dict with 'type' field.
        registry: Registry to use for building.
 
    Returns:
        Instantiated object.
    """
    return registry.build(config)