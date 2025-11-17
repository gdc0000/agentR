"""
Gemini AI integration for the Survey-to-R Agent.

This module provides functions to detect psychological constructs using
Google's Gemini AI model.
"""

from typing import Dict, List
from types import SimpleNamespace

from .models import VariableInfo, Scale, PromptConfig

# Simple config placeholder for tests and future integration
CONFIG = SimpleNamespace(GEMINI_API_KEY=None, GEMINI_MODEL=None)


def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    """
    Detect psychological constructs using Gemini AI.
    
    This is currently a dummy implementation that groups variables by prefix.
    In a real implementation, this would call the Gemini API.
    
    Args:
        var_view: List of VariableInfo objects
        prompt_cfg: Prompt configuration
        
    Returns:
        List of Scale objects proposed by the AI
    """
    if var_view is None:
        raise TypeError("var_view must be provided")
    if prompt_cfg is None:
        raise TypeError("prompt_cfg must be provided")

    # Dummy implementation: do not call external services in tests
    # Returning [] keeps the function safe by default.
    return []