"""
Unit tests for the survey_to_r.gemini module.
"""

import pytest
from unittest.mock import patch, MagicMock
from survey_to_r.gemini import gemini_detect_scales
from survey_to_r.models import VariableInfo, PromptConfig, Scale


def test_gemini_detect_scales_fallback():
    """Test fallback when API fails or no key provided."""
    variables = [
        VariableInfo(name="item_1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
        VariableInfo(name="item_2", label="Item 2", item_text="Second item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt", temperature=0.5, top_p=0.8)
    
    # Call without API key -> should trigger fallback (ValueError caught)
    with patch.dict('os.environ', {}, clear=True):
        result = gemini_detect_scales(variables, prompt_cfg)
    
    # Fallback groups by prefix (here "item")
    assert len(result) == 1
    assert result[0].name == "Item"
    assert result[0].items == ["item_1", "item_2"]


@patch('survey_to_r.gemini.genai')
def test_gemini_detect_scales_gemini_success(mock_genai):
    """Test gemini_detect_scales with mocked Gemini API."""
    # Mock the genai module
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"scales": [{"name": "Test Scale", "items": ["item1", "item2"], "confidence": 0.9}]}'
    
    mock_genai.GenerativeModel.return_value = mock_model
    mock_model.generate_content.return_value = mock_response
    
    variables = [
        VariableInfo(name="item1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg, provider="gemini", api_key="test_key")
    
    assert len(result) == 1
    assert result[0].name == "Test Scale"
    assert result[0].items == ["item1", "item2"]
    assert result[0].confidence == 0.9


@patch('survey_to_r.gemini.OpenAI')
def test_gemini_detect_scales_openrouter_success(mock_openai):
    """Test gemini_detect_scales with mocked OpenRouter API."""
    # Mock the openai client
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = '{"scales": [{"name": "OpenRouter Scale", "items": ["or1", "or2"], "confidence": 0.8}]}'
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    mock_openai.return_value = mock_client
    
    variables = [
        VariableInfo(name="or1", label="Item 1", item_text="First item", missing_pct=0.0, type="numeric"),
    ]
    
    prompt_cfg = PromptConfig(system_prompt="Test prompt")
    
    result = gemini_detect_scales(variables, prompt_cfg, provider="openrouter", api_key="test_key")
    
    assert len(result) == 1
    assert result[0].name == "OpenRouter Scale"
    assert result[0].items == ["or1", "or2"]
    assert result[0].confidence == 0.8