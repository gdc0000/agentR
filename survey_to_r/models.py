"""
Data models for the Survey-to-R Agent.

This module contains the data classes used throughout the application
to represent survey variables, scales, configuration options, and prompts.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class VariableInfo:
    """Represents metadata about a survey variable."""
    name: str
    label: Optional[str]
    item_text: Optional[str]
    missing_pct: float
    type: Literal["numeric", "ordinal", "string"]


@dataclass
class Scale:
    """Represents a psychological construct scale with its items."""
    name: str
    items: List[str]
    confidence: float = 1.0
    note: Optional[str] = None


@dataclass
class PromptConfig:
    """Configuration for AI prompt generation."""
    system_prompt: str = "Group survey items into psychological constructs."
    temperature: float = 0.2
    top_p: float = 0.9


@dataclass
class Options:
    """Analysis options for the R syntax generation."""
    include_efa: bool = True
    missing_strategy: Literal["listwise", "pairwise", "mean_scale"] = "listwise"
    correlation_type: Literal["pearson", "spearman", "polychoric"] = "pearson"
    reverse_threshold: float = 0.05