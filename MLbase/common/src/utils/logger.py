"""Logging module.
 
Singleton pattern for unified logging across the framework.
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
 
 
class Logger:
    """Singleton logger for the framework.
 
    Provides consistent logging format and output across all modules.
    """
 
    _instance: Optional['Logger'] = None
    _initialized: bool = False
 
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
 
    def __new__(cls, *args, **kwargs):
        """Ensure singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
 
    def __init__(self,
                 name: str = "ml_framework",
                 level: int = logging.INFO,
                 log_dir: Optional[Union[str, Path]] = None,
                 console_output: bool = True,
                 file_output: bool = True,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """Initialize logger.
 
        Args:
            name: Logger name.
            level: Logging level.
            log_dir: Directory for log files.
            console_output: Whether to output to console.
            file_output: Whether to output to file.
            max_bytes: Max bytes per log file.
            backup_count: Number of backup files to keep.
        """
        if Logger._initialized:
            return
 
        self.name = name
        self.level = level
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.console_output = console_output
        self.file_output = file_output
        self.max_bytes = max_bytes
        self.backup_count = backup_count
 
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False
 
        # Clear existing handlers
        self._logger.handlers.clear()
 
        # Setup handlers
        self._setup_handlers()
 
        Logger._initialized = True
 
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
 
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
 
        # File handler
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{self.name}_{timestamp}.log"
 
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
 
            self._log_file = log_file
 
    def _get_caller_info(self) -> tuple:
        """Get caller file and line info."""
        import inspect
        frame = inspect.currentframe()
        # Go up the stack to find the caller
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals.get('__name__') != __name__:
                filename = os.path.basename(frame.f_code.co_filename)
                lineno = frame.f_lineno
                return filename, lineno
        return "unknown", 0
 
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self._logger.debug(msg)
 
    def info(self, msg: str) -> None:
        """Log info message."""
        self._logger.info(msg)
 
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self._logger.warning(msg)
 
    def error(self, msg: str) -> None:
        """Log error message."""
        self._logger.error(msg)
 
    def critical(self, msg: str) -> None:
        """Log critical message."""
        self._logger.critical(msg)
 
    def exception(self, msg: str) -> None:
        """Log exception with traceback."""
        self._logger.exception(msg)
 
    def set_level(self, level: int) -> None:
        """Set logging level."""
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
 
    def add_handler(self, handler: logging.Handler) -> None:
        """Add custom handler."""
        self._logger.addHandler(handler)
 
    def get_logger(self) -> logging.Logger:
        """Get underlying logger instance."""
        return self._logger
 
 
# Global logger instance
_logger_instance: Optional[Logger] = None
 
 
def init_logger(name: str = "ml_framework",
                level: int = logging.INFO,
                log_dir: Optional[Union[str, Path]] = None,
                console_output: bool = True,
                file_output: bool = True) -> Logger:
    """Initialize global logger.
 
    Args:
        name: Logger name.
        level: Logging level.
        log_dir: Log directory.
        console_output: Console output flag.
        file_output: File output flag.
 
    Returns:
        Logger instance.
    """
    global _logger_instance
    _logger_instance = Logger(
        name=name,
        level=level,
        log_dir=log_dir,
        console_output=console_output,
        file_output=file_output
    )
    return _logger_instance
 
 
def get_logger() -> Logger:
    """Get global logger instance.
 
    Returns:
        Logger instance.
 
    Raises:
        RuntimeError: If logger not initialized.
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance
 
 
# Convenience functions
def debug(msg: str) -> None:
    """Log debug message."""
    get_logger().debug(msg)
 
 
def info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)
 
 
def warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)
 
 
def error(msg: str) -> None:
    """Log error message."""
    get_logger().error(msg)
 
 
def critical(msg: str) -> None:
    """Log critical message."""
    get_logger().critical(msg)
 
 
def exception(msg: str) -> None:
    """Log exception with traceback."""
    get_logger().exception(msg)