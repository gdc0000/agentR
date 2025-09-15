"""
Configuration management for the Survey-to-R Agent.

This module handles application configuration settings, including
default values, environment variable overrides, and validation.
"""

import os
from typing import Dict, Any, Optional


class Config:
    """Application configuration class."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {
            "log_file": os.getenv("SURVEY_TO_R_LOG_FILE", "session_log.jsonl"),
            "output_dir": os.getenv("SURVEY_TO_R_OUTPUT_DIR", "outputs"),
            "default_prompt": os.getenv("SURVEY_TO_R_DEFAULT_PROMPT", 
                                       "Group survey items into psychological constructs."),
            "default_temperature": float(os.getenv("SURVEY_TO_R_TEMPERATURE", "0.2")),
            "default_top_p": float(os.getenv("SURVEY_TO_R_TOP_P", "0.9")),
            "max_file_size_mb": int(os.getenv("SURVEY_TO_R_MAX_FILE_SIZE", "50")),
            "enable_logging": os.getenv("SURVEY_TO_R_ENABLE_LOGGING", "true").lower() == "true",
        }
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def validate(self) -> bool:
        """Validate configuration values."""
        # Check if output directory is writable
        output_dir = self.get("output_dir")
        if output_dir and not os.access(output_dir, os.W_OK):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                return False
        
        # Validate numeric values
        if not 0 <= self.get("default_temperature") <= 1:
            return False
        if not 0 <= self.get("default_top_p") <= 1:
            return False
        if self.get("max_file_size_mb") <= 0:
            return False
            
        return True


# Global configuration instance
config = Config()


def load_config_from_file(file_path: str) -> bool:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        with open(file_path, 'r') as f:
            file_config = json.load(f)
        config._config.update(file_config)
        return config.validate()
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False