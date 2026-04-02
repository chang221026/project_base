"""Integration tests for module-level interactions.

This directory contains integration tests that verify:
- Module interactions work correctly
- Data flows correctly between modules
- Interface contracts are respected

Test Organization
----------------
Tests are organized by interaction type:

- test_data_pipeline.py           # DataFacade module interactions
  - test_fetcher_to_pipeline      # fetcher output -> processor input
  - test_pipeline_to_builder      # pipeline output -> builder input

- test_training_pipeline.py       # TrainingFacade interactions
  - test_facade_to_algorithm       # facade -> algorithm construction

- test_module_interactions.py     # Cross-module interactions
  - test_trainer_to_data_facade    # Trainer -> DataFacade
  - test_trainer_to_training_facade

- test_hooks_integration.py        # Hook integrations
  - test_hook_chain_execution

- test_component_discovery.py     # Component discovery
  - test_registry_build_from_config

Running Integration Tests
--------------------------
Run all integration tests:
    python -m pytest tests/integration/ -v

Run specific interaction:
    python -m pytest tests/integration/test_data_pipeline.py -v

Expected Behavior
-----------------
These tests verify that when modules interact:
- Data is passed correctly
- Errors propagate correctly
- State is shared correctly
"""