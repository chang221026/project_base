"""Unit tests for distributed communication.

Migrated from tests/ut/test_utils_distributed_comm_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDistributedComm:
    """Test distributed communication functionality."""

    def test_distributed_manager_exists(self):
        """Test distributed manager exists."""
        from utils.distributed_comm import DistributedManager
        assert DistributedManager is not None