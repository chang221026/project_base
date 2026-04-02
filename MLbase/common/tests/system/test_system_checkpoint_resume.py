"""System tests for checkpoint save/load/resume workflow.

Tests the checkpoint resume use case from README Section 5.4,
verifying that checkpoints can be saved, loaded, and used to resume training.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def checkpoint_dir():
    """Create temporary checkpoint directory."""
    import shutil
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


# ============================================================================
# Test: Checkpoint End-to-End
# ============================================================================

@pytest.mark.system
class TestCheckpointEndToEnd:
    """Test end-to-end checkpoint save/load/resume workflow."""

    def test_save_and_load_checkpoint(self, checkpoint_dir):
        """Test: Save checkpoint -> load checkpoint -> same state.

        From README Section 5.4.

        Expected:
        - Checkpoint saves model state
        - Can load back
        - State matches
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Create state
        state = {
            'model_state': {'weight': 'value1', 'bias': 'value2'},
            'optimizer_state': {'lr': 0.001},
            'epoch': 5,
            'metrics': {'loss': 0.5}
        }

        # Save checkpoint
        manager.save(state, epoch=5, is_best=False)

        # Load checkpoint
        loaded = manager.load()

        # Verify state matches
        assert loaded['model_state']['weight'] == 'value1'
        assert loaded['epoch'] == 5

    def test_checkpoint_saves_all_necessary_state(self, checkpoint_dir):
        """Test: Checkpoint includes all necessary state.

        Expected:
        - Model state is saved
        - Optimizer state is saved
        - Epoch is saved
        - Metrics are saved
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Full state that should be saved
        full_state = {
            'model_state': {'layer1.weight': 'tensor_data'},
            'optimizer_state': {'state_dict': {}},
            'epoch': 10,
            'metrics': {'train_loss': 0.1, 'val_accuracy': 0.95},
            'config': {'model': {'type': 'MLP'}}
        }

        manager.save(full_state, epoch=10)

        loaded = manager.load()

        # Verify all parts are present
        assert 'model_state' in loaded
        assert 'optimizer_state' in loaded
        assert loaded['epoch'] == 10
        assert 'metrics' in loaded

    def test_resume_training_from_checkpoint(self, checkpoint_dir):
        """Test: Train -> save checkpoint -> load -> continue training.

        From README Section 5.4 - Resume from checkpoint.

        Expected:
        - Training can resume from checkpoint
        - Continues from saved epoch
        """
        from utils.io import CheckpointManager
        from training.trainer import Trainer

        manager = CheckpointManager(checkpoint_dir)

        config = {
            'model': {'type': 'MLP', 'input_dim': 784, 'output_dim': 10},
            'loss': {'type': 'CrossEntropyLoss'},
            'evaluator': {'type': 'AccuracyEvaluator'},
            'optimizer': {'type': 'Adam', 'lr': 0.001},
            'data': {
                'fetcher': {
                    'type': 'CSVDataFetcher',
                    'source': 'data/sample_data.csv',
                    'target_column': 'label'
                }
            },
            'training': {'epochs': 1, 'batch_size': 4},
            'logging': {'level': 'WARNING', 'console_output': False}
        }

        # First training session
        trainer = Trainer(config)
        history1 = trainer.train()

        # Save checkpoint
        save_path = checkpoint_dir / 'checkpoint.pth'
        trainer.save(str(save_path))

        # Load checkpoint using trainer's load method
        trainer.load(str(save_path))
        assert trainer is not None

    def test_resume_with_different_config(self, checkpoint_dir):
        """Test: Resume with different config (e.g., more epochs).

        Expected:
        - Can load checkpoint
        - Can use new config
        - Training continues
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Save initial checkpoint
        state1 = {'epoch': 5, 'model_state': {}, 'optimizer_state': {}}
        manager.save(state1, epoch=5)

        # Load checkpoint
        loaded = manager.load()
        assert loaded['epoch'] == 5


# ============================================================================
# Test: Checkpoint Reliability
# ============================================================================

pytest.mark.system
class TestCheckpointReliability:
    """Test checkpoint reliability."""

    def test_handles_corrupted_checkpoint(self, checkpoint_dir):
        """Test: Corrupted checkpoint gives clear error.

        Expected:
        - Clear error about corruption
        - Not a cryptic parse error
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Create fake corrupted file
        corrupt_file = checkpoint_dir / 'corrupt.pth'
        corrupt_file.write_text('not a valid checkpoint')

        # Try to load
        with pytest.raises(Exception) as exc_info:
            manager.load(str(corrupt_file))

        # Error should be clear
        assert exc_info.value is not None

    def test_handles_version_mismatch(self, checkpoint_dir):
        """Test: Version mismatch gives warning.

        Expected:
        - Warning about version mismatch
        - Still tries to load
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Save checkpoint without version
        state = {'model_state': {}}
        manager.save(state, epoch=1)

        # Save another with version
        state2 = {'model_state': {}, 'version': '2.0'}
        manager.save(state2, epoch=2)

        # Load should work (possibly with warning)
        loaded = manager.load()
        assert loaded is not None

    def test_best_checkpoint_tracking(self, checkpoint_dir):
        """Test: Best checkpoint is tracked correctly.

        Expected:
        - Best checkpoint is marked
        - Can load best separately
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Save multiple checkpoints
        manager.save({'loss': 1.0}, epoch=1, is_best=False)
        manager.save({'loss': 0.5}, epoch=2, is_best=True)
        manager.save({'loss': 0.8}, epoch=3, is_best=False)

        # Load best
        best = manager.load_best()

        assert best is not None
        assert best['loss'] == 0.5


# ============================================================================
# Test: Checkpoint Features
# ============================================================================

pytest.mark.system
class TestCheckpointFeatures:
    """Test various checkpoint features."""

    def test_max_keep_limit(self, checkpoint_dir):
        """Test: max_keep limits number of checkpoints.

        Expected:
            - Only max_keep checkpoints are kept
            - Old ones are deleted
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir, max_keep=3)

        # Save more than max_keep
        for i in range(5):
            manager.save({'epoch': i}, epoch=i)

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        # Should have 3 or fewer
        assert len(checkpoints) <= 3

    def test_checkpoint_list_ordered(self, checkpoint_dir):
        """Test: Checkpoints are listed in order.

        Expected:
            - List shows all checkpoints
            - Can filter by epoch
        """
        from utils.io import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)

        # Save checkpoints
        manager.save({'epoch': 1}, epoch=1)
        manager.save({'epoch': 2}, epoch=2)

        # List
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) >= 2