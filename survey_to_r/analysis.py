"""
Analysis functions for the Survey-to-R Agent.

This module contains functions for summarizing variables and detecting
reverse-scored items in survey data.
"""

from typing import Any, Dict, List, Set

from .models import VariableInfo, Scale


def summarize_variables(clean_meta: Dict) -> List[VariableInfo]:
    """
    Summarize variables from cleaned metadata.
    
    Args:
        clean_meta: Cleaned metadata dictionary from sanitize_metadata
        
    Returns:
        List of VariableInfo objects representing each variable
    """
    var_labels = clean_meta.get("var_labels", {})
    if isinstance(var_labels, list):  # fallback defensive
        names = clean_meta.get("var_names", list(range(len(var_labels))))
        var_labels = {n: l for n, l in zip(names, var_labels)}

    dropped = set(clean_meta.get("dropped", []))

    return [
        VariableInfo(
            name=name,
            label=label,
            item_text=None,
            missing_pct=0.0,
            type="numeric",
        )
        for name, label in var_labels.items()
        if name not in dropped
    ]


def detect_reverse_items(scales_confirmed: List[Scale], df: Any) -> Dict[str, bool]:
    """
    Detect reverse-scored items based on correlation with scale scores.
    
    Args:
        scales_confirmed: List of confirmed scales
        df: Cleaned DataFrame containing the data
        
    Returns:
        Dictionary mapping item names to reverse flag (True if reverse-scored)
    """
    import pandas as pd
    
    rev_map: Dict[str, bool] = {}
    for sc in scales_confirmed:
        if len(sc.items) < 2:
            continue
        try:
            # Select only numeric columns for the scale items
            sub = df[sc.items].select_dtypes(include=['number']).dropna()
            if sub.empty or len(sub.columns) < 2:  # Need at least 2 items for correlation
                continue
            scale_score = sub.mean(axis=1)
            for item in sub.columns:  # Iterate only over numeric columns
                corr = sub[item].corr(scale_score)
                rev_map[item] = corr is not None and corr < 0
        except Exception as e:
            # Log error and continue with other scales
            print(f"Error processing scale {sc.name}: {e}")
            continue
    return rev_map