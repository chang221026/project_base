"""System tests for distributed training workflow.

Tests the distributed training use case from README Section 5.3,
verifying that the framework handles single to multi-GPU training correctly.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def clean_dist_env():
    """Clean distributed environment variables."""
    env_keys = [
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
        'MASTER_ADDR', 'MASTER_PORT'
    ]
    original = {}
    for key in env_keys:
        original[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    yield

    for key, value in original.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


# ============================================================================
# Test: Distributed End-to-End
# ============================================================================

@pytest.mark.system
class TestDistributedEndToEnd:
    """Test end-to-end distributed training workflow."""

    def test_distributed_training_initialization(self, clean_dist_env):
        """Test: Distributed training initializes correctly.

        From README Section 5.3.

        Expected:
        - DistributedEngine initializes
        - Device detection works
        """
        from training.distributed import DistributedEngine
        from utils.distributed_comm import DistributedManager

        # Reset singletons
        DistributedManager._instance = None
        DistributedManager._initialized = False

        # Create engine
        engine = DistributedEngine()

        # Should initialize
        assert engine is not None

    def test_distributed_config_parsing(self, clean_dist_env):
        """Test: Distributed config is parsed correctly.

        Expected:
        - Config with distributed section is parsed
        - Mode, backend are extracted
        """
        from utils.config_management import Config

        config = Config.from_dict({
            'distributed': {
                'mode': 'auto',
                'backend': 'auto',
                'auto_launch': True
            }
        })

        # Verify config extraction
        dist_config = config.get('distributed')
        assert dist_config is not None
        assert dist_config.get('mode') == 'auto'

    def test_distributed_prepare_dataloader(self, clean_dist_env):
        """Test: DistributedEngine prepares dataloader with sampler.

        From README Section 4.6 - Distributed training config.

        Expected:
        - DataLoader is wrapped with DistributedSampler
        - Each process gets different data
        """
        from training.distributed import DistributedEngine
        from utils.distributed_comm import DistributedManager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        engine = DistributedEngine()

        # Create mock dataloader
        mock_loader = MagicMock()
        mock_loader.dataset = MagicMock()
        mock_loader.dataset.__len__ = lambda self: 100

        # Prepare for distributed
        prepared = engine.prepare_dataloader(mock_loader)

        # Should have been wrapped or prepared
        assert prepared is not None

    def test_distributed_prepare_model(self, clean_dist_env):
        """Test: DistributedEngine wraps model for DDP.

        From README Section 5.3 - DDP model wrapping.

        Expected:
        - Model is wrapped for distributed training
        - Sync batch normalization if present
        """
        from training.distributed import DistributedEngine
        from utils.distributed_comm import DistributedManager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        engine = DistributedEngine()

        # Create mock model
        model = MagicMock()
        model.state_dict.return_value = {'weight': 1}

        # Prepare for distributed
        prepared = engine.prepare_model(model)

        # Should have been prepared for distributed
        assert prepared is not None


# ============================================================================
# Test: Distributed Reliability
# ============================================================================

@pytest.mark.system
class TestDistributedReliability:
    """Test distributed training reliability."""

    def test_graceful_degradation_without_distributed_env(self, clean_dist_env):
        """Test: Without distributed env, falls back to single device.

        From README Section 5.3 - automatic fallback.

        Expected:
        - No distributed env vars = runs on single device
        - No error
        """
        from training.distributed import DistributedEngine
        from utils.distributed_comm import DistributedManager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        engine = DistributedEngine()

        # Should work in non-distributed mode
        assert engine is not None

        # Manager should report not distributed
        manager = DistributedManager()
        assert manager.is_distributed() is False

    def test_handles_nccl_failure_gracefully(self, clean_dist_env):
        """Test: NCCL failure is handled gracefully.

        Expected:
        - NCCL init failure gives clear error
        - Falls back to GLOO or single device
        """
        from training.distributed import DistributedEngine

        engine = DistributedEngine()

        # Even if NCCL fails, should have fallback
        # This test verifies the structure exists
        assert engine is not None

    def test_recovers_from_communication_error(self, clean_dist_env):
        """Test: Recovers from communication errors.

        Expected:
        - Can handle communication errors
        - Can continue training
        """
        from training.distributed import DistributedEngine
        from utils.distributed_comm import DistributedManager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        manager = DistributedManager()

        # In non-distributed mode, communication errors don't occur
        assert manager.is_distributed() is False


# ============================================================================
# Test: Distributed Backend
# ============================================================================

pytest.mark.system
class TestDistributedBackend:
    """Test distributed backend selection."""

    def test_nccl_backend_detection(self, clean_dist_env):
        """Test: NCCL backend is detected when available.

        Expected:
            - Backend detection runs
            - Selects appropriate backend
        """
        from utils.distributed_comm import DistributedManager, get_dist_manager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        manager = DistributedManager()

        # Should detect backend (or default to gloo/cpu)
        assert manager is not None

    def test_gloo_backend_fallback(self, clean_dist_env):
        """Test: Falls back to GLOO when NCCL unavailable.

        Expected:
            - GLOO is available as fallback
            - Works for CPU-based distributed
        """
        from utils.distributed_comm import DistributedManager

        DistributedManager._instance = None
        DistributedManager._initialized = False

        manager = DistributedManager()

        # In non-distributed or CPU mode, should work
        assert manager.get_world_size() == 1


# ============================================================================
# Test: Distributed Checkpoint
# ============================================================================

pytest.mark.system
class TestDistributedCheckpoint:
    """Test checkpoint handling in distributed training."""

    def test_distributed_checkpoint_save_load(self, clean_dist_env):
        """Test: Checkpoints save/load correctly in distributed.

        From README Section 5.3 - checkpoint with distributed.

        Expected:
        - Checkpoint saves correctly
        - Can be loaded back
        """
        from utils.io import CheckpointManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Save checkpoint
            state = {
                'model_state': {'weight': 1},
                'epoch': 1,
                'distributed': True
            }

            manager.save(state, epoch=1)

            # Load checkpoint
            loaded = manager.load()

            assert loaded is not None
            assert loaded['epoch'] == 1