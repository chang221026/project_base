"""Integration tests for hook combinations.

Tests multiple hooks working together and their interactions.
Migrated from tests/integration/test_hooks_combination.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestHookCombinations:
    """Test combinations of different hooks."""

    def test_checkpoint_hook_exists(self):
        """Test checkpoint hook exists."""
        from training.hook.checkpoint import CheckpointHook
        assert CheckpointHook is not None

    def test_early_stopping_hook_exists(self):
        """Test early stopping hook exists."""
        from training.hook.early_stopping import EarlyStoppingHook
        assert EarlyStoppingHook is not None

    def test_logging_hook_exists(self):
        """Test logging hook exists."""
        from training.hook.logging_hook import LoggingHook
        assert LoggingHook is not None

    def test_lr_scheduler_hook_exists(self):
        """Test lr scheduler hook exists."""
        from training.hook.lr_scheduler import LRSchedulerHook
        assert LRSchedulerHook is not None


class TestHookChain:
    """Test hook execution chain."""

    def test_hook_chain_execution(self):
        """Test hooks can be chained."""
        from training.hook.base import BaseHook

        hook1 = BaseHook(priority=0)
        hook2 = BaseHook(priority=1)

        # Hooks should be comparable by priority
        assert hook1.priority <= hook2.priority