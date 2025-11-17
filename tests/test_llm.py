"""
Unit tests for the survey_to_r.llm module.
"""

import pytest
from unittest.mock import patch, MagicMock
from survey_to_r.llm import gemini_detect_scales
from survey_to_r.models import VariableInfo, PromptConfig, Scale


def test_llm_detect_scales_no_key():
    """Test llm_detect_scales when no API key is configured."""
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
        VariableInfo(name="item2", label="Item 2", item_text="Second item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt", temperature=0.5, top_p=0.8)
    
    # Call the function
    result = gemini_detect_scales(variables, prompt_cfg)
    
    # Should return empty list if no API key
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []


def test_llm_detect_scales_with_empty_variables():
    """Test gemini_detect_scales with empty variables list."""
    variables = []
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []


def test_llm_detect_scales_with_none_variables():
    """Test gemini_detect_scales with None variables."""
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    with pytest.raises(TypeError):
        gemini_detect_scales(None, prompt_cfg)


def test_llm_detect_scales_with_none_prompt_config():
    """Test gemini_detect_scales with None prompt config."""
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    with pytest.raises(TypeError):
        gemini_detect_scales(variables, None)


@patch('survey_to_r.llm.config')
@patch('openai.OpenAI')
def test_llm_detect_scales_with_mock_api(mock_openai, mock_config):
    """Test gemini_detect_scales with mocked API calls."""
    # Mock the configuration
    mock_config.get.return_value = "test_api_key"  # for openrouter_api_key
    mock_config.get.side_effect = lambda key, default=None: "test_model" if key == "openrouter_model" else default
    
    # Mock the openai client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '[{"name": "Test Scale", "items": ["item1", "item2"]}]'
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value.__enter__.return_value = mock_client
    
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
        VariableInfo(name="item2", label="Item 2", item_text="Second item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg)
    assert len(result) == 1
    assert result[0].name == "Test Scale"
    assert result[0].items == ["item1", "item2"]


def test_llm_detect_scales_prompt_config_values():
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
    
    # With mock to avoid real call
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '[]'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value.__enter__.return_value = mock_client
        
        result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []


@patch('survey_to_r.llm.config')
def test_llm_detect_scales_no_api_key(mock_config):
    """Test gemini_detect_scales when no API key is configured."""
    mock_config.get.return_value = None  # no api key
    
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg)
    assert result == []