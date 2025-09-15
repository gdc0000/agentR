"""
Unit tests for the survey_to_r.utils module.
"""

import os
import tempfile
import pytest
import json
from unittest.mock import patch, MagicMock
from survey_to_r.utils import setup_logging, log_session, orchestrate_pipeline
from survey_to_r.models import VariableInfo, Scale, Options, PromptConfig


def test_setup_logging():
    """Test setup_logging function creates log directory and file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = os.path.join(tmp_dir, "logs")
        log_file = os.path.join(log_dir, "test.log")
        
        # Call setup_logging
        setup_logging(log_file)
        
        # Verify log directory was created
        assert os.path.exists(log_dir)
        
        # Verify log file was created
        assert os.path.exists(log_file)


def test_setup_logging_existing_dir():
    """Test setup_logging with existing log directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = os.path.join(tmp_dir, "logs")
        os.makedirs(log_dir)  # Create directory first
        log_file = os.path.join(log_dir, "test.log")
        
        # Call setup_logging
        setup_logging(log_file)
        
        # Verify log file was created
        assert os.path.exists(log_file)


@patch('survey_to_r.utils.config')
def test_log_session(mock_config):
    """Test log_session function writes JSONL events correctly."""
    mock_config.get.return_value = "test_log.jsonl"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_log = os.path.join(tmp_dir, "test_log.jsonl")
        mock_config.get.return_value = test_log
        
        # Test logging an event
        test_event = {"event": "test", "message": "test message"}
        log_session(test_event)
        
        # Verify log file was created and contains the event
        assert os.path.exists(test_log)
        
        with open(test_log, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            logged_event = json.loads(lines[0])
            assert logged_event["event"] == "test"
            assert logged_event["message"] == "test message"
            assert "timestamp" in logged_event
            assert logged_event["level"] == "info"


@patch('survey_to_r.utils.config')
def test_log_session_with_existing_fields(mock_config):
    """Test log_session preserves existing timestamp and level fields."""
    mock_config.get.return_value = "test_log.jsonl"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_log = os.path.join(tmp_dir, "test_log.jsonl")
        mock_config.get.return_value = test_log
        
        # Test logging an event with existing timestamp and level
        test_event = {
            "event": "test",
            "timestamp": "2023-01-01T00:00:00",
            "level": "warning"
        }
        log_session(test_event)
        
        # Verify log file was created and contains the event
        assert os.path.exists(test_log)
        
        with open(test_log, 'r') as f:
            lines = f.readlines()
            logged_event = json.loads(lines[0])
            assert logged_event["timestamp"] == "2023-01-01T00:00:00"
            assert logged_event["level"] == "warning"


def test_orchestrate_pipeline_invalid_file():
    """Test orchestrate_pipeline with invalid file path."""
    with pytest.raises(FileNotFoundError):
        orchestrate_pipeline("non_existent_file.sav", Options(), PromptConfig())


@patch('survey_to_r.utils.load_sav')
@patch('survey_to_r.utils.sanitize_metadata')
@patch('survey_to_r.utils.summarize_variables')
@patch('survey_to_r.utils.detect_reverse_items')
@patch('survey_to_r.utils.gemini_detect_scales')
@patch('survey_to_r.utils.build_r_syntax')
@patch('survey_to_r.utils.write_r_file')
def test_orchestrate_pipeline_success(mock_write_r_file, mock_build_r_syntax, mock_gemini_detect_scales,
                                     mock_detect_reverse_items, mock_summarize_variables,
                                     mock_sanitize_metadata, mock_load_sav):
    """Test orchestrate_pipeline successful execution with mocked dependencies."""
    # Mock the dependencies
    mock_load_sav.return_value = (MagicMock(), {"variable_labels": {"var1": "Label"}})
    mock_sanitize_metadata.return_value = {"variable_labels": {"var1": "Label"}}
    mock_summarize_variables.return_value = [
        VariableInfo(name="var1", label="Label", item_text="Item", missing_pct=0.0, type="numeric")
    ]
    mock_detect_reverse_items.return_value = {}
    mock_gemini_detect_scales.return_value = [
        Scale(name="Test Scale", items=["var1"], confidence=0.8)
    ]
    mock_build_r_syntax.return_value = "# Test R code"
    
    # Call orchestrate_pipeline
    result = orchestrate_pipeline("test.sav", Options(), PromptConfig())
    
    # Verify all mocks were called
    mock_load_sav.assert_called_once_with("test.sav")
    mock_sanitize_metadata.assert_called_once()
    mock_summarize_variables.assert_called_once()
    mock_detect_reverse_items.assert_called_once()
    mock_gemini_detect_scales.assert_called_once()
    mock_build_r_syntax.assert_called_once()
    mock_write_r_file.assert_called_once()
    
    # Verify result is the R syntax
    assert result == "# Test R code"


@patch('survey_to_r.utils.load_sav')
@patch('survey_to_r.utils.sanitize_metadata')
@patch('survey_to_r.utils.summarize_variables')
@patch('survey_to_r.utils.detect_reverse_items')
@patch('survey_to_r.utils.gemini_detect_scales')
@patch('survey_to_r.utils.build_r_syntax')
@patch('survey_to_r.utils.write_r_file')
def test_orchestrate_pipeline_no_scales(mock_write_r_file, mock_build_r_syntax, mock_gemini_detect_scales,
                                       mock_detect_reverse_items, mock_summarize_variables,
                                       mock_sanitize_metadata, mock_load_sav):
    """Test orchestrate_pipeline when no scales are detected."""
    # Mock the dependencies
    mock_load_sav.return_value = (MagicMock(), {"variable_labels": {"var1": "Label"}})
    mock_sanitize_metadata.return_value = {"variable_labels": {"var1": "Label"}}
    mock_summarize_variables.return_value = [
        VariableInfo(name="var1", label="Label", item_text="Item", missing_pct=0.0, type="numeric")
    ]
    mock_detect_reverse_items.return_value = {}
    mock_gemini_detect_scales.return_value = []  # No scales detected
    mock_build_r_syntax.return_value = "# Basic R code"
    
    # Call orchestrate_pipeline
    result = orchestrate_pipeline("test.sav", Options(), PromptConfig())
    
    # Verify gemini_detect_scales was called
    mock_gemini_detect_scales.assert_called_once()
    
    # Verify build_r_syntax was called with empty scales
    mock_build_r_syntax.assert_called_once()
    call_args = mock_build_r_syntax.call_args
    assert call_args[0][1] == []  # scales argument should be empty list
    
    # Verify result is the basic R syntax
    assert result == "# Basic R code"


@patch('survey_to_r.utils.load_sav')
def test_orchestrate_pipeline_load_sav_error(mock_load_sav):
    """Test orchestrate_pipeline handles load_sav error."""
    mock_load_sav.side_effect = ValueError("Invalid file format")
    
    with pytest.raises(ValueError, match="Invalid file format"):
        orchestrate_pipeline("invalid.sav", Options(), PromptConfig())


@patch('survey_to_r.utils.load_sav')
@patch('survey_to_r.utils.sanitize_metadata')
@patch('survey_to_r.utils.summarize_variables')
@patch('survey_to_r.utils.detect_reverse_items')
@patch('survey_to_r.utils.gemini_detect_scales')
@patch('survey_to_r.utils.build_r_syntax')
@patch('survey_to_r.utils.write_r_file')
def test_orchestrate_pipeline_write_error(mock_write_r_file, mock_build_r_syntax, mock_gemini_detect_scales,
                                         mock_detect_reverse_items, mock_summarize_variables,
                                         mock_sanitize_metadata, mock_load_sav):
    """Test orchestrate_pipeline handles write_r_file error."""
    # Mock the dependencies up to write_r_file
    mock_load_sav.return_value = (MagicMock(), {"variable_labels": {"var1": "Label"}})
    mock_sanitize_metadata.return_value = {"variable_labels": {"var1": "Label"}}
    mock_summarize_variables.return_value = [
        VariableInfo(name="var1", label="Label", item_text="Item", missing_pct=0.0, type="numeric")
    ]
    mock_detect_reverse_items.return_value = {}
    mock_gemini_detect_scales.return_value = [
        Scale(name="Test Scale", items=["var1"], confidence=0.8)
    ]
    mock_build_r_syntax.return_value = "# Test R code"
    mock_write_r_file.side_effect = PermissionError("Permission denied")
    
    # Call orchestrate_pipeline - should raise the error
    with pytest.raises(PermissionError, match="Permission denied"):
        orchestrate_pipeline("test.sav", Options(), PromptConfig())
    
    # Verify all mocks were called except write_r_file which failed
    mock_load_sav.assert_called_once()
    mock_sanitize_metadata.assert_called_once()
    mock_summarize_variables.assert_called_once()
    mock_detect_reverse_items.assert_called_once()
    mock_gemini_detect_scales.assert_called_once()
    mock_build_r_syntax.assert_called_once()
    mock_write_r_file.assert_called_once()