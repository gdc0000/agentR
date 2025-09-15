"""
Utility functions for the Survey-to-R Agent.

This module contains logging and orchestration utilities.
"""

import json
import pathlib
from datetime import datetime
from typing import Dict, List, Optional

from .models import VariableInfo, Scale, Options, PromptConfig


def setup_logging(log_file_path: str) -> None:
    """
    Set up logging by ensuring the log directory exists.
    
    Args:
        log_file_path: Path to the log file
    """
    log_path = pathlib.Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def log_session(event: Dict) -> None:
    """
    Append a single JSON event (one-line) to persistent log.
    
    Args:
        event: Dictionary containing event data
    """
    from .config import config
    
    log_file_path = config.get("log_file", "session_log.jsonl")
    setup_logging(log_file_path)
    
    log_path = pathlib.Path(log_file_path)
    event.setdefault("timestamp", datetime.now().isoformat())
    event.setdefault("level", "info")
    
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def orchestrate_pipeline(
    file_path: str,
    prompt_config: Optional[PromptConfig] = None,
    options: Optional[Options] = None
) -> str:
    """
    Orchestrate the complete pipeline from file loading to R syntax generation.
    
    Args:
        file_path: Path to the SPSS file
        prompt_config: Optional prompt configuration for AI detection
        options: Optional analysis options
        
    Returns:
        Generated R script
        
    Raises:
        RuntimeError: If any step in the pipeline fails
    """
    from .io import load_sav, sanitize_metadata, write_r_file
    from .analysis import summarize_variables, detect_reverse_items
    from .gemini import gemini_detect_scales
    from .r_syntax import build_r_syntax
    
    # Set default configurations if not provided
    if prompt_config is None:
        prompt_config = PromptConfig(
            system_prompt="Group survey items into psychological constructs."
        )
    
    if options is None:
        options = Options()
    
    try:
        # Step 1: Load and sanitize data
        df, meta = load_sav(file_path)
        clean_df, clean_meta = sanitize_metadata(df, meta)
        
        # Step 2: Summarize variables
        var_view = summarize_variables(clean_meta)
        
        # Step 3: Detect scales using AI
        scales_proposed = gemini_detect_scales(var_view, prompt_config)
        
        # For orchestration, we'll use all proposed scales
        confirmed_scales = scales_proposed
        
        # Step 4: Detect reverse items
        rev_map = detect_reverse_items(confirmed_scales, clean_df)
        
        # Step 5: Build R syntax
        r_script = build_r_syntax(
            sav_path=file_path,
            scales=confirmed_scales,
            rev_map=rev_map,
            opts=options
        )
        
        # Step 6: Write R file
        output_path = write_r_file(r_script)
        
        # Log the session
        log_session({
            "event": "pipeline_completed",
            "file_path": file_path,
            "scales_detected": len(confirmed_scales),
            "reverse_items": sum(rev_map.values()),
            "output_path": output_path
        })
        
        return r_script
        
    except Exception as e:
        log_session({
            "event": "pipeline_error",
            "file_path": file_path,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        raise RuntimeError(f"Pipeline orchestration failed: {e}") from e