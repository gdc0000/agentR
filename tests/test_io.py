"""
Unit tests for the survey_to_r.io module.
"""

import os
import tempfile
import pytest
from unittest.mock import mock_open, patch
import pandas as pd
from survey_to_r.io import load_sav, sanitize_metadata, write_r_file


def test_load_sav_file_not_found():
    """Test load_sav raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_sav("non_existent_file.sav")


def test_load_sav_invalid_file():
    """Test load_sav raises ValueError for invalid file content."""
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp.write(b"invalid content")
        tmp_path = tmp.name
    
    try:
        with pytest.raises(ValueError):
            load_sav(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_sanitize_metadata():
    """Test sanitize_metadata function with various inputs."""
    # Test with empty metadata
    assert sanitize_metadata({}) == {}
    
    # Test with typical metadata
    meta = {
        "variable_labels": {"var1": "Label 1", "var2": "Label 2"},
        "value_labels": {"var1": {1: "Yes", 2: "No"}},
        "missing_ranges": {"var1": [99, 999]},
        "variable_display_width": {"var1": 10, "var2": 15}
    }
    
    result = sanitize_metadata(meta)
    assert "variable_labels" in result
    assert "value_labels" in result
    assert "missing_ranges" in result
    assert "variable_display_width" in result


def test_sanitize_metadata_with_none():
    """Test sanitize_metadata handles None values gracefully."""
    meta = {
        "variable_labels": None,
        "value_labels": {"var1": {1: "Yes"}},
        "missing_ranges": None,
        "variable_display_width": {"var1": 10}
    }
    
    result = sanitize_metadata(meta)
    assert result["variable_labels"] == {}
    assert result["value_labels"] == {"var1": {1: "Yes"}}
    assert result["missing_ranges"] == {}
    assert result["variable_display_width"] == {"var1": 10}


def test_write_r_file_success():
    """Test write_r_file successfully writes content to file."""
    with tempfile.NamedTemporaryFile(suffix=".R", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        content = "# Test R content\nprint('Hello World')"
        write_r_file(tmp_path, content)
        
        # Verify file was written
        with open(tmp_path, 'r') as f:
            written_content = f.read()
        assert written_content == content
    finally:
        os.unlink(tmp_path)


def test_write_r_file_directory_not_exists():
    """Test write_r_file creates directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_path = os.path.join(tmp_dir, "subdir", "test.R")
        content = "# Test R content"
        
        write_r_file(nested_path, content)
        
        # Verify file was created
        assert os.path.exists(nested_path)
        with open(nested_path, 'r') as f:
            assert f.read() == content


def test_write_r_file_permission_error():
    """Test write_r_file handles permission errors gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a file that's read-only to simulate permission error
        test_file = os.path.join(tmp_dir, "test.R")
        with open(test_file, 'w') as f:
            f.write("existing content")
        os.chmod(test_file, 0o444)  # Read-only
        
        try:
            with pytest.raises(PermissionError):
                write_r_file(test_file, "new content")
        finally:
            os.chmod(test_file, 0o644)  # Restore permissions
            os.unlink(test_file)