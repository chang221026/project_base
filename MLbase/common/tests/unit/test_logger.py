"""Unit tests for Logger.

Migrated from tests/ut/test_utils_logger_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestLogger:
    """Test Logger functionality."""

    def test_logger_exists(self):
        """Test Logger exists."""
        from utils.logger import Logger
        assert Logger is not None