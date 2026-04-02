"""Unit tests for DeviceManager.

Migrated from tests/ut/test_utils_device_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDeviceManager:
    """Test DeviceManager functionality."""

    def test_device_manager_exists(self):
        """Test DeviceManager exists."""
        from utils.device_management import DeviceManager
        assert DeviceManager is not None