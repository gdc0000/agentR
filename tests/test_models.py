"""
Unit tests for the survey_to_r.models module.
"""

import pytest
from survey_to_r.models import VariableInfo, Scale, PromptConfig, Options


def test_variable_info_creation():
    """Test VariableInfo data class creation and properties."""
    var_info = VariableInfo(
        name="test_var",
        label="Test Variable",
        item_text="This is a test variable",
        missing_pct=5.0,
        type="numeric"
    )
    
    assert var_info.name == "test_var"
    assert var_info.label == "Test Variable"
    assert var_info.item_text == "This is a test variable"
    assert var_info.missing_pct == 5.0
    assert var_info.type == "numeric"


def test_scale_creation():
    """Test Scale data class creation and properties."""
    scale = Scale(
        name="Test Scale",
        items=["item1", "item2", "item3"],
        confidence=0.8,
        note="This is a test scale"
    )
    
    assert scale.name == "Test Scale"
    assert scale.items == ["item1", "item2", "item3"]
    assert scale.confidence == 0.8
    assert scale.note == "This is a test scale"


def test_prompt_config_creation():
    """Test PromptConfig data class creation and properties."""
    prompt_cfg = PromptConfig(
        system_prompt="Test system prompt",
        temperature=0.5,
        top_p=0.8
    )
    
    assert prompt_cfg.system_prompt == "Test system prompt"
    assert prompt_cfg.temperature == 0.5
    assert prompt_cfg.top_p == 0.8


def test_options_creation():
    """Test Options data class creation and properties."""
    options = Options(
        include_efa=False,
        missing_strategy="pairwise",
        correlation_type="spearman",
        reverse_threshold=0.1
    )
    
    assert options.include_efa is False
    assert options.missing_strategy == "pairwise"
    assert options.correlation_type == "spearman"
    assert options.reverse_threshold == 0.1


def test_scale_default_values():
    """Test Scale data class default values."""
    scale = Scale(name="Test Scale", items=["item1"])
    
    assert scale.confidence == 1.0
    assert scale.note is None


def test_prompt_config_default_values():
    """Test PromptConfig data class default values."""
    prompt_cfg = PromptConfig(system_prompt="Test")
    
    assert prompt_cfg.temperature == 0.2
    assert prompt_cfg.top_p == 0.9


def test_options_default_values():
    """Test Options data class default values."""
    options = Options()
    
    assert options.include_efa is True
    assert options.missing_strategy == "listwise"
    assert options.correlation_type == "pearson"
    assert options.reverse_threshold == 0.05