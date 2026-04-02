"""Exception handling module.
 
Provides unified exception hierarchy for the framework.
"""
from typing import Optional, Any
import traceback
 
 
class MLFrameworkError(Exception):
    """Base exception for all framework errors."""
 
    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize exception.
 
        Args:
            message: Error message.
            details: Optional error details dictionary.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
 
    def __str__(self) -> str:
        """String representation."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message
 
 
# Configuration errors
class ConfigError(MLFrameworkError):
    """Configuration related errors."""
    pass
 
 
class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass
 
 
class ConfigParseError(ConfigError):
    """Configuration parsing error."""
    pass
 
 
# Device errors
class DeviceError(MLFrameworkError):
    """Device related errors."""
    pass
 
 
class DeviceNotAvailableError(DeviceError):
    """Requested device not available."""
    pass
 
 
class DeviceOutOfMemoryError(DeviceError):
    """Device out of memory."""
    pass
 
 
# Model errors
class ModelError(MLFrameworkError):
    """Model related errors."""
    pass
 
 
class ModelNotFoundError(ModelError):
    """Model not found in registry."""
    pass
 
 
class ModelLoadError(ModelError):
    """Error loading model."""
    pass
 
 
class ModelSaveError(ModelError):
    """Error saving model."""
    pass
 
 
# Data errors
class DataError(MLFrameworkError):
    """Data related errors."""
    pass
 
 
class DataNotFoundError(DataError):
    """Data file not found."""
    pass
 
 
class DataLoadError(DataError):
    """Error loading data."""
    pass
 
 
class DataFormatError(DataError):
    """Invalid data format."""
    pass
 
 
class DataValidationError(DataError):
    """Data validation failed."""
    pass
 
 
# Training errors
class TrainingError(MLFrameworkError):
    """Training related errors."""
    pass
 
 
class TrainingInterruptedError(TrainingError):
    """Training was interrupted."""
    pass
 
 
class TrainingConfigError(TrainingError):
    """Training configuration error."""
    pass
 
 
class CheckpointError(TrainingError):
    """Checkpoint save/load error."""
    pass
 
 
# Distributed errors
class DistributedError(MLFrameworkError):
    """Distributed training related errors."""
    pass
 
 
class DistributedInitError(DistributedError):
    """Distributed initialization error."""
    pass
 
 
class DistributedCommunicationError(DistributedError):
    """Distributed communication error."""
    pass
 
 
# Registry errors
class RegistryError(MLFrameworkError):
    """Registry related errors."""
    pass
 
 
class ComponentNotFoundError(RegistryError):
    """Component not found in registry."""
    pass
 
 
class ComponentAlreadyExistsError(RegistryError):
    """Component already registered."""
    pass
 
 
# IO errors
class MLIOError(MLFrameworkError):
    """IO related errors."""
    pass
 
 
class MLFileNotFoundError(MLIOError):
    """File not found."""
    pass
 
 
class FileAccessError(MLIOError):
    """File access error."""
    pass
 
 
class SerializationError(MLIOError):
    """Serialization error."""
    pass
 
 
class ExceptionHandler:
    """Centralized exception handler.
 
    Provides consistent error handling and logging.
    """
 
    def __init__(self, logger=None):
        """Initialize exception handler.
 
        Args:
            logger: Optional logger instance.
        """
        self.logger = logger
 
    def handle(self, exception: Exception, re_raise: bool = False) -> dict:
        """Handle an exception.
 
        Args:
            exception: Exception to handle.
            re_raise: Whether to re-raise the exception.
 
        Returns:
            Error information dictionary.
        """
        error_info = {
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc()
        }
 
        # Add framework-specific details
        if isinstance(exception, MLFrameworkError):
            error_info['details'] = exception.details
 
        # Log the error
        if self.logger:
            if isinstance(exception, (MLFrameworkError,)):
                self.logger.error(f"Framework error: {error_info['message']}")
            else:
                self.logger.error(f"Unexpected error: {error_info['message']}")
            self.logger.debug(f"Traceback:\n{error_info['traceback']}")
 
        if re_raise:
            raise exception
 
        return error_info
 
    def wrap(self, func, *args, **kwargs):
        """Wrap a function with exception handling.
 
        Args:
            func: Function to wrap.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
 
        Returns:
            Function result or error info.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle(e, re_raise=False)
 
 
def handle_exception(exception: Exception,
                     logger=None,
                     re_raise: bool = False) -> dict:
    """Convenience function to handle exceptions.
 
    Args:
        exception: Exception to handle.
        logger: Optional logger.
        re_raise: Whether to re-raise.
 
    Returns:
        Error information.
    """
    handler = ExceptionHandler(logger)
    return handler.handle(exception, re_raise)
 
 
# Backwards compatibility aliases (deprecated, will be removed in future version)
# Use MLIOError and MLFileNotFoundError instead to avoid shadowing builtins
IOError = MLIOError  # Deprecated: shadows builtin IOError
FileNotFoundError = MLFileNotFoundError  # Deprecated: shadows builtin FileNotFoundError