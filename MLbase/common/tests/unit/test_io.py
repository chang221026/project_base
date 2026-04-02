"""Unit tests for IO utilities.

Migrated from tests/ut/test_utils_io_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestIOUtils:
    """Test IO utilities functionality."""

    def test_io_utils_exists(self):
        """Test IO utilities exist."""
        from utils import io
        assert io is not None