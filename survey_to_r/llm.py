"""
LLM integration for the Survey-to-R Agent using OpenRouter API.

This module provides functions to detect psychological constructs using
OpenRouter API (OpenAI-compatible).
"""

from typing import Dict, List
import json
import openai
from types import SimpleNamespace

from .models import VariableInfo, Scale, PromptConfig
from .config import config


def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    """
    Detect psychological constructs using OpenRouter API.
    
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

    api_key = config.get("openrouter_api_key")
    if not api_key:
        # No API key, fallback to empty
        return []

    model = config.get("openrouter_model", "openai/gpt-4o-mini")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Construct prompt
    variables_text = "\n".join([
        f"- {v.name}: {v.label or v.item_text or 'No description'}"
        for v in var_view
    ])
    
    user_prompt = f"""
{prompt_cfg.system_prompt}

Survey variables:
{variables_text}

Group these into psychological constructs/scales. Respond ONLY with valid JSON array of objects:
[
  {{"name": "Construct Name", "items": ["var_name1", "var_name2"]}},
  ...
]
Each scale should have 2+ items. Use exact variable names.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=prompt_cfg.temperature,
            top_p=prompt_cfg.top_p,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from content if needed (in case of extra text)
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        scales_data = json.loads(content)
        
        scales = []
        for s in scales_data:
            if isinstance(s, dict) and "name" in s and "items" in s and len(s["items"]) >= 2:
                scales.append(Scale(
                    name=s["name"],
                    items=s["items"],
                    confidence=1.0  # Default confidence
                ))
        
        return scales
        
    except json.JSONDecodeError:
        # Invalid JSON response
        return []
    except Exception as e:
        # Any other error, fallback
        print(f"LLM error: {e}")
        return []