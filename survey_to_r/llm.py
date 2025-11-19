"""
LLM integration for the Survey-to-R Agent using OpenRouter API or Google Gemini.

This module provides functions to detect psychological constructs using
OpenRouter API (OpenAI-compatible) or Google's Gemini API.
"""

from typing import Dict, List, Optional
import json
import openai
from types import SimpleNamespace
import google.generativeai as genai

from .models import VariableInfo, Scale, PromptConfig
from .config import config


def gemini_detect_scales(
    var_view: List[VariableInfo], 
    prompt_cfg: PromptConfig,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> List[Scale]:
    """
    Detect psychological constructs using OpenRouter API or Gemini.
    
    Args:
        var_view: List of VariableInfo objects
        prompt_cfg: Prompt configuration
        provider: "gemini" or "openrouter"
        api_key: API key for the provider
        model_name: Model name to use
        
    Returns:
        List of Scale objects proposed by the AI
    """
    if var_view is None:
        raise TypeError("var_view must be provided")
    if prompt_cfg is None:
        raise TypeError("prompt_cfg must be provided")

    # Use provided API key or fallback to config
    if not api_key:
        if provider == "openrouter":
            api_key = config.get("openrouter_api_key")
        else:
            api_key = config.get("gemini_api_key")
    
    if not api_key:
        # No API key, fallback to empty
        print(f"No API key provided for {provider}")
        return []

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
        content = ""
        
        if provider == "openrouter":
            model = model_name or config.get("openrouter_model", "openai/gpt-4o-mini")
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=prompt_cfg.temperature,
                top_p=prompt_cfg.top_p,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            
        elif provider == "gemini":
            model = model_name or "gemini-pro"
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            
            generation_config = genai.types.GenerationConfig(
                temperature=prompt_cfg.temperature,
                top_p=prompt_cfg.top_p,
                max_output_tokens=1000,
            )
            
            response = model_instance.generate_content(
                user_prompt,
                generation_config=generation_config
            )
            content = response.text.strip()
            
        else:
            print(f"Unknown provider: {provider}")
            return []

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
        print(f"JSON Decode Error. Content: {content}")
        return []
    except Exception as e:
        # Any other error, fallback
        print(f"LLM error: {e}")
        return []