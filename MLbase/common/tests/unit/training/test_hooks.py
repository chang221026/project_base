"""Unit tests for hooks interface.

Migrated from tests/ut/test_training_hook_*.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestBaseHook:
    """Test BaseHook functionality."""

    def test_base_hook_exists(self):
        """Test BaseHook exists."""
        from training.hook.base import BaseHook
        assert BaseHook is not None

    def test_hook_callbacks(self):
        """Test hook callbacks are callable."""
        from training.hook.base import BaseHook

        hook = BaseHook(priority=0)

        # All callbacks should exist
        assert hasattr(hook, 'on_train_start')
        assert hasattr(hook, 'on_train_end')
        assert hasattr(hook, 'on_epoch_start')
        assert hasattr(hook, 'on_epoch_end')


class TestCheckpointHook:
    """Test CheckpointHook functionality."""

    def test_checkpoint_hook_exists(self):
        """Test CheckpointHook exists."""
        from training.hook.checkpoint import CheckpointHook
        assert CheckpointHook is not None


class TestEarlyStoppingHook:
    """Test EarlyStoppingHook functionality."""

    def test_early_stopping_hook_exists(self):
        """Test EarlyStoppingHook exists."""
        from training.hook.early_stopping import EarlyStoppingHook
        assert EarlyStoppingHook is not None


class TestLoggingHook:
    """Test LoggingHook functionality."""

    def test_logging_hook_exists(self):
        """Test LoggingHook exists."""
        from training.hook.logging_hook import LoggingHook
        assert LoggingHook is not None


class TestLRSchedulerHook:
    """Test LRSchedulerHook functionality."""

    def test_lr_scheduler_hook_exists(self):
        """Test LRSchedulerHook exists."""
        from training.hook.lr_scheduler import LRSchedulerHook
        assert LRSchedulerHook is not None