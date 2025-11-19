"""
Gemini AI and OpenRouter integration for the Survey-to-R Agent.

This module provides functions to detect psychological constructs using
Google's Gemini AI model or OpenRouter.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from openai import OpenAI

from .models import VariableInfo, Scale, PromptConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gemini_detect_scales(
    var_view: List[VariableInfo], 
    prompt_cfg: PromptConfig,
    provider: str = "gemini",
    api_key: str = "",
    model_name: str = "gemini-pro"
) -> List[Scale]:
    """
    Detect psychological constructs using an LLM.
    
    Args:
        var_view: List of VariableInfo objects
        prompt_cfg: Prompt configuration
        provider: "gemini" or "openrouter"
        api_key: API Key for the provider
        model_name: Model name to use
        
    Returns:
        List of Scale objects proposed by the AI
    """
    
    # Construct the prompt
    variables_text = "\n".join([
        f"- {v.name}: {v.label} (Text: {v.item_text})" 
        for v in var_view
    ])
    
    user_prompt = f"""
    Analyze the following survey variables and group them into psychological constructs (scales).
    
    Variables:
    {variables_text}
    
    Output strictly valid JSON in the following format:
    {{
        "scales": [
            {{
                "name": "Construct Name",
                "items": ["var1", "var2"],
                "confidence": 0.9,
                "note": "Reasoning..."
            }}
        ]
    }}
    
    IMPORTANT: Use ONLY the variable names listed above. Do not invent new variables.
    """
    
    try:
        if provider == "openrouter":
            response_text = _call_openrouter(user_prompt, prompt_cfg, api_key, model_name)
        else:
            response_text = _call_gemini(user_prompt, prompt_cfg, api_key, model_name)
            
        # Parse JSON
        # Clean up code blocks if present
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(response_text)
        scales_data = data.get("scales", [])
        
        scales = []
        for s in scales_data:
            scales.append(Scale(
                name=s["name"],
                items=s["items"],
                confidence=s.get("confidence", 0.5),
                note=s.get("note")
            ))
            
        return scales
        
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        # Fallback to dummy implementation if API fails or key is missing
        logger.info("Falling back to dummy implementation")
        clusters: dict[str, list[str]] = {}
        for v in var_view:
            prefix = v.name.split("_")[0] if "_" in v.name else "misc"
            clusters.setdefault(prefix, []).append(v.name)
        return [Scale(name=k.title(), items=items, confidence=0.3, note=f"Fallback ({str(e)})") for k, items in clusters.items()]


def _call_gemini(prompt: str, cfg: PromptConfig, api_key: str, model_name: str) -> str:
    """Call Google Gemini API."""
    # Use provided key or env var
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API Key not provided")
        
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )
    )
    
    return response.text


def _call_openrouter(prompt: str, cfg: PromptConfig, api_key: str, model_name: str) -> str:
    """Call OpenRouter API."""
    if not api_key:
        raise ValueError("OpenRouter API Key not provided")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    
    return completion.choices[0].message.content