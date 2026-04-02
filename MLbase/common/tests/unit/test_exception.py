"""Unit tests for exceptions.

Migrated from tests/ut/test_utils_exception_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestExceptions:
    """Test exception classes."""

    def test_exception_handler_exists(self):
        """Test ExceptionHandler exists."""
        from utils.exception import ExceptionHandler
        assert ExceptionHandler is not None