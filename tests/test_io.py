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


def test_load_sav_tempfile_cleanup(monkeypatch):
    """Ensure load_sav removes temporary files created for bytes/file-like inputs."""
    from unittest.mock import MagicMock
    import survey_to_r.io as io_mod

    # Mock pyreadstat to succeed
    monkeypatch.setattr(io_mod, 'pyreadstat', MagicMock(read_sav=MagicMock(return_value=([], {}))))

    # Mock NamedTemporaryFile to create a predictable temp file name and record cleanup
    class DummyTemp:
        def __init__(self, *args, **kwargs):
            self.name = os.path.join(tempfile.gettempdir(), "dummy_temp.sav")
            self.file = None
        def __enter__(self):
            self.file = open(self.name, 'wb')
            self.file.write(b"content")
            self.file.flush()
            return self
        def __exit__(self, exc_type, exc, tb):
            if self.file:
                self.file.close()
            return False
        def write(self, data):
            if self.file:
                self.file.write(data)

    monkeypatch.setattr(io_mod.tempfile, 'NamedTemporaryFile', DummyTemp)
    # Track os.remove calls
    removed = []

    def fake_remove(path):
        removed.append(path)

    monkeypatch.setattr(io_mod.os, 'remove', fake_remove)

    # Run load_sav using bytes input
    df, meta = io_mod.load_sav(b"abc")

    # Ensure temp file was removed
    assert any(p.endswith("dummy_temp.sav") for p in removed)


def test_load_sav_enforces_max_size(monkeypatch):
    """Large bytes input exceeding max_file_size_mb should be rejected."""
    import survey_to_r.io as io_mod
    from survey_to_r.config import config

    # Set a small max size
    config.set("max_file_size_mb", 0)

    # Mock pyreadstat to avoid actual read after size check
    with patch('survey_to_r.io.pyreadstat') as mock_pyreadstat:
        with pytest.raises(ValueError, match="exceeds maximum"):
            io_mod.load_sav(b"12345")


def test_sanitize_metadata():
    """Test sanitize_metadata function with various inputs."""
    # Test with empty metadata
    empty_df = pd.DataFrame()
    result = sanitize_metadata(empty_df, {})
    assert result[1] == {'dropped': [], 'var_names': []}

    # Test with typical metadata
    df = pd.DataFrame({"var1": [1,2], "var2": [3,4]})
    meta = {
        "variable_labels": {"var1": "Label 1", "var2": "Label 2"},
        "value_labels": {"var1": {1: "Yes", 2: "No"}},
        "missing_ranges": {"var1": [99, 999]},
        "variable_display_width": {"var1": 10, "var2": 15}
    }
    
    result = sanitize_metadata(df, meta)
    assert "variable_labels" in result[1]
    assert "value_labels" in result[1]
    assert "missing_ranges" in result[1]
    assert "variable_display_width" in result[1]


def test_sanitize_metadata_with_none():
    """Test sanitize_metadata handles None values gracefully."""
    df = pd.DataFrame({"var1": [1,2]})
    meta = {
        "variable_labels": None,
        "value_labels": {"var1": {1: "Yes"}},
        "missing_ranges": None,
        "variable_display_width": {"var1": 10}
    }
    
    result = sanitize_metadata(df, meta)
    assert result[1]["variable_labels"] == {}
    assert result[1]["value_labels"] == {"var1": {1: "Yes"}}
    assert result[1]["missing_ranges"] == {}
    assert result[1]["variable_display_width"] == {"var1": 10}


def test_write_r_file_success():
    """Test write_r_file successfully writes content to file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        parent_dir = os.path.dirname(tmp_dir)
        # Set the root output dir to parent to make tmp_dir a subdir
        from survey_to_r.config import config
        config.set("root_output_dir", parent_dir)
        out_dir = tmp_dir
    
        content = "# Test R content\nprint('Hello World')"
        output_path = write_r_file(content, out_dir=out_dir)

        # Verify file was written
        rfile = os.path.join(out_dir, "analysis.R")
        with open(rfile, 'r') as f:
            written_content = f.read()
        assert written_content == content
        assert output_path == rfile


def test_write_r_file_directory_not_exists():
    """Test write_r_file creates directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        parent_dir = os.path.dirname(tmp_dir)
        from survey_to_r.config import config
        config.set("root_output_dir", parent_dir)
        nested_dir = os.path.join(tmp_dir, "subdir")
        content = "# Test R content"

        output_path = write_r_file(content, out_dir=nested_dir)

        rfile = os.path.join(nested_dir, "analysis.R")
        # Verify file was created
        assert os.path.exists(rfile)
        with open(rfile, 'r') as f:
            assert f.read() == content
        assert output_path == rfile


def test_write_r_file_permission_error():
    """Test write_r_file handles permission errors gracefully."""
    if os.name == 'nt':
        pytest.skip("Permission simulation with chmod not supported on Windows")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        parent_dir = os.path.dirname(tmp_dir)
        from survey_to_r.config import config
        config.set("root_output_dir", parent_dir)
        
        # Create read-only dir to simulate permission error
        restricted_dir = os.path.join(tmp_dir, "restricted")
        os.makedirs(restricted_dir)
        os.chmod(restricted_dir, 0o444)  # Read-only
        
        try:
            with pytest.raises(PermissionError):
                write_r_file("new content", out_dir=restricted_dir)
        finally:
            os.chmod(restricted_dir, 0o755)