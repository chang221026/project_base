"""Unit tests for EVALUATORS registry.

Migrated from tests/ut/test_lib_evaluator_*.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestEvaluatorsRegistry:
    """Test EVALUATORS registry functionality."""

    def test_evaluators_registry_exists(self):
        """Test EVALUATORS registry exists."""
        from lib.evaluator import EVALUATORS
        assert EVALUATORS is not None

    def test_base_evaluator_exists(self):
        """Test BaseEvaluator exists."""
        from lib.evaluator.base import BaseEvaluator
        assert BaseEvaluator is not None

    def test_accuracy_registration(self):
        """Test AccuracyEvaluator is registered."""
        from lib.evaluator import EVALUATORS

        acc = EVALUATORS.get('AccuracyEvaluator')
        assert acc is not None

    def test_build_accuracy(self):
        """Test building AccuracyEvaluator from config."""
        from lib.evaluator import EVALUATORS

        evaluator = EVALUATORS.build({'type': 'AccuracyEvaluator'})
        assert evaluator is not None

    def test_accuracy_evaluate(self):
        """Test AccuracyEvaluator evaluate."""
        import numpy as np
        from lib.evaluator import EVALUATORS

        evaluator = EVALUATORS.build({'type': 'AccuracyEvaluator'})

        predictions = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 1, 1])

        metrics = evaluator.evaluate(predictions, targets)
        assert 'accuracy_top1' in metrics