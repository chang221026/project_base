"""System tests for framework-level end-to-end functionality.

This directory contains system tests that verify:
- End-to-end user scenarios work correctly
- The framework is reliable under failure conditions
- The framework scales to production workloads
- The framework is usable and provides good UX

Test Organization
-----------------
Test files are organized by user scenario (matching README Section 5):

- test_system_trainer_workflow.py    # Section 5.1: Config-driven training
- test_system_custom_extensibility.py # Section 5.2: Custom extensibility
- test_system_distributed_training.py # Section 5.3: Distributed training
- test_system_checkpoint_resume.py   # Section 5.4: Checkpoint resume
- test_system_rl_training.py         # Section 5.5: RL training

Additional system quality tests:

- test_system_reliability.py        # Fault tolerance, error handling
- test_system_scalability.py        # Performance at scale
- test_system_usability.py          # API usability, error messages

Test Naming Convention
----------------------
Tests follow this naming pattern:

    TestSystemScenarioName
        - test_scenario_xxx_end_to_end    # Main user flow
        - test_scenario_xxx_reliability  # Error handling
        - test_scenario_xxx_edge_case    # Edge conditions

Running System Tests
--------------------
Run all system tests:
    python -m pytest tests/system/ -v

Run specific scenario:
    python -m pytest tests/system/ -k trainer -v
    python -m pytest tests/system/ -k distributed -v

Run reliability tests:
    python -m pytest tests/system/ -v -k reliability

Run usability tests:
    python -m pytest tests/system/ -v -k usability
"""