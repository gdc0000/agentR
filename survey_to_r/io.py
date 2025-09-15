"""
Input/Output operations for the Survey-to-R Agent.

This module handles reading SPSS (.sav) files, sanitizing metadata,
and writing R script files.
"""

import os
import tempfile
from typing import Any, Dict, Tuple, Union, BinaryIO

from .models import VariableInfo


def _ensure_pandas() -> Any:
    """Ensure pandas is available and return the pandas module."""
    try:
        import pandas as _pd  # noqa: F401
        _PANDAS_AVAILABLE = True
    except ModuleNotFoundError as e:
        _PANDAS_AVAILABLE = False if e.name == "micropip" else True
    
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required but not available in this environment.")
    
    import pandas as pd
    return pd


def load_sav(path_or_file: Union[str, os.PathLike, bytes, bytearray, BinaryIO]) -> Tuple[Any, Dict]:
    """
    Load an SPSS (.sav) file and return DataFrame with metadata.
    
    Args:
        path_or_file: Path to .sav file or file-like object
        
    Returns:
        Tuple of (DataFrame, metadata dictionary)
        
    Raises:
        RuntimeError: If pyreadstat is not installed
        TypeError: If input type is unsupported
    """
    pd = _ensure_pandas()
    
    try:
        import pyreadstat  # heavy; local import
    except ModuleNotFoundError:
        raise RuntimeError("pyreadstat is required to read .sav files but is not installed.")

    if isinstance(path_or_file, (str, os.PathLike)):
        sav_path = os.fspath(path_or_file)
        df, meta = pyreadstat.read_sav(sav_path, apply_value_formats=False)
    else:
        # Bytes or file-like → temp file
        if isinstance(path_or_file, (bytes, bytearray)):
            raw_bytes = bytes(path_or_file)
            orig_name = "bytes_input.sav"
        elif hasattr(path_or_file, "read"):
            raw_bytes = path_or_file.getvalue() if hasattr(path_or_file, "getvalue") else path_or_file.read()
            orig_name = getattr(path_or_file, "name", "uploaded_file.sav")
        else:
            raise TypeError("Unsupported input type for load_sav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name
        df, meta = pyreadstat.read_sav(temp_path, apply_value_formats=False)
        sav_path = orig_name

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    meta_dict = {
        "sav_name": sav_path,
        "var_labels": meta.column_names_to_labels,  # dict
        "var_names": meta.column_names,             # list
        "value_labels": meta.variable_value_labels,
        "var_types": meta.original_variable_types,
        "missing_ranges": meta.missing_ranges,
    }
    return df, meta_dict


def sanitize_metadata(df: Any, meta: Dict) -> Tuple[Any, Dict]:
    """
    Clean and sanitize metadata by removing problematic columns.
    
    Args:
        df: Raw DataFrame from SPSS file
        meta: Metadata dictionary from SPSS file
        
    Returns:
        Tuple of (cleaned DataFrame, cleaned metadata)
    """
    pd = _ensure_pandas()

    drop_cols: list[str] = []
    for col in df.columns:
        if (df[col].dtype == "object" and df[col].nunique() > 50) or col.lower() in {"id", "participant_id", "timestamp", "date"}:
            drop_cols.append(col)

    clean_df = df.drop(columns=drop_cols)

    for col, ranges in meta.get("missing_ranges", {}).items():
        if col not in clean_df.columns:
            continue
        for lo, hi in ranges:
            clean_df.loc[clean_df[col].between(lo, hi), col] = pd.NA

    # Prune metadata
    clean_meta = meta.copy()
    clean_meta["dropped"] = drop_cols
    for key in ("var_labels", "var_types", "missing_ranges"):
        if key in clean_meta and isinstance(clean_meta[key], dict):
            clean_meta[key] = {k: v for k, v in clean_meta[key].items() if k not in drop_cols}
    clean_meta["var_names"] = [n for n in meta.get("var_names", []) if n not in drop_cols]

    return clean_df, clean_meta


def write_r_file(r_script: str, out_dir: str = "outputs") -> str:
    """
    Write R script to a file.
    
    Args:
        r_script: R script content
        out_dir: Output directory path
        
    Returns:
        Path to the written R file
    """
    import pathlib
    
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    r_path = pathlib.Path(out_dir) / "analysis.R"
    r_path.write_text(r_script, encoding="utf-8")
    return str(r_path)