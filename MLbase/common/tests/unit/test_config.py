"""Unit tests for Config interface.

Migrated from tests/ut/test_utils_config.py
"""
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestConfig:
    """Test Config interface functionality."""

    def test_config_from_dict(self):
        """Test config from dictionary."""
        from utils.config_management import Config

        config = Config.from_dict({'key': 'value'})
        assert config.get('key') == 'value'

    def test_config_get_nested(self):
        """Test nested config get."""
        from utils.config_management import Config

        config = Config.from_dict({
            'a': {'b': {'c': 1}}
        })
        assert config.get('a.b.c') == 1

    def test_config_set(self):
        """Test config set value."""
        from utils.config_management import Config

        config = Config.from_dict({'key': 'value'})
        config.set('key', 'new_value')
        assert config.get('key') == 'new_value'

    def test_load_yaml(self, temp_yaml_file):
        """Test loading YAML config."""
        from utils.config_management import load_config

        config = load_config(str(temp_yaml_file))
        assert config is not None

    def test_load_json(self, temp_json_file):
        """Test loading JSON config."""
        from utils.config_management import load_config

        config = load_config(str(temp_json_file))
        assert config is not None


@pytest.fixture
def temp_yaml_file():
    """Create temp YAML file."""
    import tempfile
    from pathlib import Path
    import yaml

    fd, path = tempfile.mkstemp(suffix='.yaml')
    os.close(fd)  # Close file descriptor first
    Path(path).unlink()

    config = {'model': {'type': 'test'}}
    with open(path, 'w') as f:
        yaml.dump(config, f)

    yield Path(path)
    Path(path).unlink()


@pytest.fixture
def temp_json_file():
    """Create temp JSON file."""
    import tempfile
    from pathlib import Path
    import json

    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)  # Close file descriptor first
    Path(path).unlink()

    config = {'model': {'type': 'test'}}
    with open(path, 'w') as f:
        json.dump(config, f)

    yield Path(path)
    Path(path).unlink()