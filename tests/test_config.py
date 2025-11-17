"""
Unit tests for the survey_to_r.config module.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from survey_to_r.config import Config, load_config_from_file, config


def test_config_initialization():
    """Test Config class initialization with default values."""
    cfg = Config()
    
    # Test default values
    assert cfg.get("log_file") == "session_log.jsonl"
    assert cfg.get("output_dir") == "outputs"
    assert cfg.get("default_prompt") == "Group survey items into psychological constructs."
    assert cfg.get("default_temperature") == 0.2
    assert cfg.get("default_top_p") == 0.9
    assert cfg.get("max_file_size_mb") == 50
    assert cfg.get("enable_logging") is True
    # New keys
    assert cfg.get("root_output_dir") == "outputs"
    assert cfg.get("mask_file_names") is True


def test_config_get_set():
    """Test Config get and set methods."""
    cfg = Config()
    
    # Test setting and getting values
    cfg.set("test_key", "test_value")
    assert cfg.get("test_key") == "test_value"
    
    # Test getting non-existent key with default
    assert cfg.get("non_existent_key", "default") == "default"
    assert cfg.get("non_existent_key") is None


@patch.dict(os.environ, {
    "SURVEY_TO_R_LOG_FILE": "custom_log.jsonl",
    "SURVEY_TO_R_OUTPUT_DIR": "custom_outputs",
    "SURVEY_TO_R_DEFAULT_PROMPT": "Custom prompt",
    "SURVEY_TO_R_TEMPERATURE": "0.5",
    "SURVEY_TO_R_TOP_P": "0.8",
    "SURVEY_TO_R_MAX_FILE_SIZE": "100",
    "SURVEY_TO_R_ENABLE_LOGGING": "false"
})
def test_config_environment_variables():
    """Test Config reads environment variables correctly."""
    cfg = Config()
    
    assert cfg.get("log_file") == "custom_log.jsonl"
    assert cfg.get("output_dir") == "custom_outputs"
    assert cfg.get("default_prompt") == "Custom prompt"
    assert cfg.get("default_temperature") == 0.5
    assert cfg.get("default_top_p") == 0.8
    assert cfg.get("max_file_size_mb") == 100
    assert cfg.get("enable_logging") is False


def test_config_validation_success():
    """Test Config validation with valid values."""
    cfg = Config()
    assert cfg.validate() is True


@patch.dict(os.environ, {
    "SURVEY_TO_R_TEMPERATURE": "1.5",  # Invalid: > 1
    "SURVEY_TO_R_TOP_P": "-0.1",       # Invalid: < 0
    "SURVEY_TO_R_MAX_FILE_SIZE": "-10" # Invalid: <= 0
})
def test_config_validation_failure():
    """Test Config validation with invalid values."""
    cfg = Config()
    assert cfg.validate() is False


def test_config_validation_output_dir_permissions():
    """Test Config validation handles output directory permissions."""
    if os.name == 'nt':  # Skip on Windows as chmod doesn't simulate read-only effectively
        pytest.skip("Permission simulation with chmod not supported on Windows")
    
    cfg = Config()
    
    # Test with non-writable directory (simulate permission error)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a subdirectory with no write permissions
        restricted_dir = os.path.join(tmp_dir, "restricted")
        os.makedirs(restricted_dir)
        os.chmod(restricted_dir, 0o444)  # Read-only
        
        cfg.set("output_dir", restricted_dir)
        
        # Validation should fail due to permission issues
        assert cfg.validate() is False
        
        # Restore permissions for cleanup
        os.chmod(restricted_dir, 0o755)


def test_load_config_from_file_success():
    """Test load_config_from_file with valid JSON file."""
    from importlib import reload
    import survey_to_r.config
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write('{"log_file": "file_log.jsonl", "output_dir": "file_outputs"}')
        tmp_path = tmp.name
    
    try:
        result = load_config_from_file(tmp_path)
        assert result is True
        
        # Verify config was updated
        assert config.get("log_file") == "file_log.jsonl"
        assert config.get("output_dir") == "file_outputs"
    finally:
        os.unlink(tmp_path)
    
    # Reload to reset global config for other tests
    reload(survey_to_r.config)


def test_load_config_from_file_invalid_json():
    """Test load_config_from_file with invalid JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write('invalid json content')
        tmp_path = tmp.name
    
    try:
        result = load_config_from_file(tmp_path)
        assert result is False
    finally:
        os.unlink(tmp_path)


def test_load_config_from_file_nonexistent():
    """Test load_config_from_file with non-existent file."""
    result = load_config_from_file("non_existent_file.json")
    assert result is False


def test_load_config_from_file_validation_failure():
    """Test load_config_from_file fails when loaded config fails validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp.write('{"default_temperature": 1.5}')  # Invalid value > 1
        tmp_path = tmp.name
    
    try:
        result = load_config_from_file(tmp_path)
        assert result is False
    finally:
        os.unlink(tmp_path)


def test_global_config_instance():
    """Test that the global config instance is properly initialized."""
    from importlib import reload
    import survey_to_r.config
    reload(survey_to_r.config)
    assert isinstance(config, Config)
    assert config.get("log_file") == "session_log.jsonl"
    assert config.get("output_dir") == "outputs"


@patch.dict(os.environ, {
    "SURVEY_TO_R_LOG_FILE": "test_log.jsonl"
})
def test_global_config_environment_override():
    """Test that global config respects environment overrides."""
    # Re-initialize config to pick up environment changes
    from importlib import reload
    import survey_to_r.config
    reload(survey_to_r.config)
    
    assert survey_to_r.config.config.get("log_file") == "test_log.jsonl"