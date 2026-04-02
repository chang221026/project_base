"""Unit tests for framework components.

This directory contains unit tests that verify:
- Interface/class implementation correctness
- Code-level functionality (line/branch coverage)
- Each component works in isolation

Test Organization
----------------
Tests are organized by layer and component:

- test_registry.py          # Registry interface
- test_config.py            # Config interface
- test_device.py            # DeviceManager interface
- test_logger.py            # Logger interface
- test_io.py                # IO interface
- test_distributed_comm.py # Distributed communication interface

lib/                       # Lib layer components
    - test_models.py        # MODELS registry
    - test_losses.py        # LOSSES registry
    - test_optimizers.py    # OPTIMIZERS registry
    - test_evaluators.py    # EVALUATORS registry
    - test_data_fetchers.py
    - test_data_processors.py

training/                 # Training layer components
    - test_algorithms.py    # Algorithm interface
    - test_hooks.py          # Hook interface
    - test_distributed_engine.py
    - test_data_pipeline.py
    - test_dataset.py

Running Unit Tests
------------------
Run all unit tests:
    python -m pytest tests/unit/ -v --cov

Run specific component:
    python -m pytest tests/unit/test_registry.py -v
    python -m pytest tests/unit/lib/test_models.py -v

Expected Coverage
------------------
Each test file should achieve:
- Line coverage: > 70%
- Branch coverage: > 60%
"""