"""
Survey-to-R Agent - A tool for converting SPSS survey data to R analysis syntax.

This package provides functionality to load SPSS (.sav) files, detect psychological
constructs using AI, and generate R syntax for statistical analysis.
"""

from .models import (
    VariableInfo,
    Scale,
    PromptConfig,
    Options
)

from .io import (
    load_sav,
    sanitize_metadata,
    write_r_file
)

from .analysis import (
    summarize_variables,
    detect_reverse_items
)

from .llm import (
    gemini_detect_scales
)

from .r_syntax import (
    build_r_syntax
)

from .utils import (
    log_session,
    orchestrate_pipeline
)

__version__ = "0.1.0"
__all__ = [
    "VariableInfo",
    "Scale",
    "PromptConfig",
    "Options",
    "load_sav",
    "sanitize_metadata",
    "summarize_variables",
    "gemini_detect_scales",
    "detect_reverse_items",
    "build_r_syntax",
    "write_r_file",
    "log_session",
    "orchestrate_pipeline"
]