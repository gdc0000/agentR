"""
Unit tests for the survey_to_r.gemini module.
"""

import pytest
from unittest.mock import patch, MagicMock
from survey_to_r.gemini import gemini_detect_scales
from survey_to_r.models import VariableInfo, PromptConfig, Scale


def test_gemini_detect_scales_dummy_implementation():
    """Test the dummy implementation of gemini_detect_scales."""
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
        VariableInfo(name="item2", label="Item 2", item_text="Second item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt", temperature=0.5, top_p=0.8)
    
    # Call the function
    result = gemini_detect_scales(variables, prompt_cfg)
    
    # Should return empty list for dummy implementation
    assert result == []


def test_gemini_detect_scales_with_empty_variables():
    """Test gemini_detect_scales with empty variables list."""
    variables = []
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []


def test_gemini_detect_scales_with_none_variables():
    """Test gemini_detect_scales with None variables."""
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    with pytest.raises(TypeError):
        gemini_detect_scales(None, prompt_cfg)


def test_gemini_detect_scales_with_none_prompt_config():
    """Test gemini_detect_scales with None prompt config."""
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    with pytest.raises(TypeError):
        gemini_detect_scales(variables, None)


@patch('survey_to_r.gemini.CONFIG')
def test_gemini_detect_scales_with_mock_api(mock_config):
    """Test gemini_detect_scales with mocked API calls."""
    # Mock the configuration
    mock_config.GEMINI_API_KEY = "test_api_key"
    mock_config.GEMINI_MODEL = "test_model"
    
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
        VariableInfo(name="item2", label="Item 2", item_text="Second item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    # This should work if the API key is set, but since we're mocking, it should proceed
    # However, the current implementation is dummy, so it will return empty list
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []  # Dummy implementation always returns empty list


def test_gemini_detect_scales_prompt_config_values():
    """Test that prompt config values are accessible."""
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(
        system_prompt="System instruction",
        temperature=0.7,
        top_p=0.95
    )
    
    # Just verify the prompt config is passed correctly (though not used in dummy)
    assert prompt_cfg.system_prompt == "System instruction"
    assert prompt_cfg.temperature == 0.7
    assert prompt_cfg.top_p == 0.95
    
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []  # Dummy implementation


@patch('survey_to_r.gemini.CONFIG')
def test_gemini_detect_scales_no_api_key(mock_config):
    """Test gemini_detect_scales when no API key is configured."""
    mock_config.GEMINI_API_KEY = None
    mock_config.GEMINI_MODEL = "test_model"
    
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    # Should still work with dummy implementation
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []