from __future__ import annotations

"""Core pipeline utilities for the Survey‑to‑R Agent.

This module implements the *minimum‑viable* versions of every function
specified in the architecture outline. They are deliberately lightweight and
kept dependency‑free except for **pandas** and **pyreadstat**.

Key change (2025‑07‑14)
----------------------
The `load_sav` function now gracefully handles *Streamlit* `UploadedFile`
objects (or any file‑like with a `.read()` method) by reading their binary
content and passing it to **pyreadstat** with `io_bytes=True`. This resolves
``TypeError: expected str, bytes or os.PathLike object, not UploadedFile``.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Literal, Tuple
import json
import os
import pathlib
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VariableInfo:
    name: str
    label: str | None
    item_text: str | None
    missing_pct: float
    type: Literal["numeric", "ordinal", "string"]


@dataclass
class Scale:
    name: str
    items: List[str]
    confidence: float = 1.0
    note: str | None = None


@dataclass
class PromptConfig:
    system_prompt: str
    temperature: float = 0.2
    top_p: float = 0.9


@dataclass
class Options:
    include_efa: bool = True
    missing_strategy: Literal["listwise", "pairwise", "mean_scale"] = "listwise"
    correlation_type: Literal["pearson", "spearman", "polychoric"] = "pearson"
    reverse_threshold: float = 0.05


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
    "orchestrate_pipeline",
]

# ---------------------------------------------------------------------------
# 1. Load .sav (robust to UploadedFile)
# ---------------------------------------------------------------------------

def load_sav(path_or_file) -> Tuple[pd.DataFrame, Dict]:
    """Read a **SPSS .sav** file from *disk path*, *bytes*, *BytesIO* **or** a
    *Streamlit* ``UploadedFile``.

    Parameters
    ----------
    path_or_file : str | os.PathLike | bytes | BytesIO | UploadedFile
        Anything returned by `st.file_uploader` or a file‑system path.

    Returns
    -------
    df  : pandas.DataFrame
        Raw data in tabular form (value formats *not* applied).
    meta : dict
        Simplified metadata: variable labels, value labels, types, missing ranges.
    """
    import pyreadstat  # lazy import

    # Case 1 – Streamlit UploadedFile or any file‑like with .read()
    if hasattr(path_or_file, "read") and not isinstance(path_or_file, (str, bytes, os.PathLike)):
        try:
            raw_bytes: bytes = path_or_file.getvalue()  # Streamlit UploadedFile
        except AttributeError:
            raw_bytes = path_or_file.read()  # generic file‑like
        df, meta = pyreadstat.read_sav(raw_bytes, apply_value_formats=False, io_bytes=True)
        sav_name = getattr(path_or_file, "name", "uploaded_file.sav")
    # Case 2 – bytes already
    elif isinstance(path_or_file, (bytes, bytearray)):
        df, meta = pyreadstat.read_sav(path_or_file, apply_value_formats=False, io_bytes=True)
        sav_name = "bytes_input.sav"
    # Case 3 – normal path on disk
    else:
        df, meta = pyreadstat.read_sav(path_or_file, apply_value_formats=False)
        sav_name = os.fspath(path_or_file)

    meta_dict = {
        "sav_name": sav_name,
        "var_labels": meta.column_labels,
        "value_labels": meta.variable_value_labels,
        "var_types": meta.original_variable_types,
        "missing_ranges": meta.missing_ranges,
    }
    return df, meta_dict


# ---------------------------------------------------------------------------
# 2. Sanitize metadata (simple placeholder)
# ---------------------------------------------------------------------------

def sanitize_metadata(df: pd.DataFrame, meta: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Drop obvious non‑numeric/non‑ordinal columns and normalise missing codes."""
    drop_cols: List[str] = []
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() > 50:
            drop_cols.append(col)
        elif col.lower() in {"id", "participant_id", "timestamp", "date"}:
            drop_cols.append(col)

    clean_df = df.drop(columns=drop_cols)

    # Replace user‑defined missings with NaN
    for col, ranges in meta.get("missing_ranges", {}).items():
        if col not in clean_df.columns:
            continue
        for lo, hi in ranges:
            clean_df.loc[clean_df[col].between(lo, hi), col] = pd.NA

    clean_meta = meta.copy()
    clean_meta["dropped"] = drop_cols
    return clean_df, clean_meta


# ---------------------------------------------------------------------------
# 3. Summarize variables (placeholder: missing_pct always 0.0)
# ---------------------------------------------------------------------------

def summarize_variables(clean_meta: Dict) -> List[VariableInfo]:
    var_labels = clean_meta.get("var_labels", {})
    view: List[VariableInfo] = []
    for name, label in var_labels.items():
        view.append(
            VariableInfo(
                name=name,
                label=label,
                item_text=None,
                missing_pct=0.0,
                type="numeric",
            )
        )
    return view


# ---------------------------------------------------------------------------
# 4. Gemini detect scales (dummy clustering by prefix)
# ---------------------------------------------------------------------------

def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    clusters: Dict[str, List[str]] = {}
    for v in var_view:
        key = v.name.split("_")[0] if "_" in v.name else "misc"
        clusters.setdefault(key, []).append(v.name)
    return [Scale(name=k.title(), items=items, confidence=0.3) for k, items in clusters.items()]


# ---------------------------------------------------------------------------
# 6. Detect reverse items (simple correlation sign test)
# ---------------------------------------------------------------------------

def detect_reverse_items(scales_confirmed: List[Scale], df: pd.DataFrame) -> Dict[str, bool]:
    rev_map: Dict[str, bool] = {}
    for sc in scales_confirmed:
        if len(sc.items) < 2:
            continue
        sub = df[sc.items].dropna()
        if sub.empty:
            continue
        scale_score = sub.mean(axis=1)
        for item in sc.items:
            corr = sub[item].corr(scale_score)
            rev_map[item] = corr is not None and corr < 0
    return rev_map


# ---------------------------------------------------------------------------
# 7. Build R syntax (core exporter)
# ---------------------------------------------------------------------------

def build_r_syntax(sav_path: str, scales: List[Scale], rev_map: Dict[str, bool], opts: Options) -> str:
    lines: List[str] = []
    L = lines.append

    L("# -------------------------------------------------------------")
    L("# Auto‑generated R syntax (Survey‑to‑R Agent)")
    L(f"# Generated: {datetime.now().isoformat()}")
    L("# -------------------------------------------------------------\n")

    # 1. Packages
    L("if (!require('pacman')) install.packages('pacman')")
    L("pacman::p_load(haven, tidyverse, psych, MBESS, GPArotation, lavaan)")

    # 2. Import
    L(f"data <- haven::read_sav('{sav_path}')\n")

    # 3. Reverse scoring
    if any(rev_map.values()):
        L("# Reverse‑score flagged items")
        for item, flag in rev_map.items():
            if flag:
                L(f"maxv <- max(data${item}, na.rm = TRUE)")
                L(f"minv <- min(data${item}, na.rm = TRUE)")
                L(f"data${item} <- maxv + minv - data${item}\n")

    # 4. Reliability per scale
    L("# Reliability analysis (Cronbach α / McDonald Ω)")
    for sc in scales:
        item_vec_name = f"items_{sc.name.lower()}"
        items_csv = ", ".join(f"'{it}'" for it in sc.items)
        L(f"{item_vec_name} <- c({items_csv})")
        L(f"alpha_{sc.name.lower()} <- psych::alpha(data[{item_vec_name}], check.keys = TRUE)")
        L(f"print(alpha_{sc.name.lower()}$total)  # Cronbach α total")
        L(f"omega_{sc.name.lower()} <- MBESS::ci.reliability(data[{item_vec_name}], type = 'omega')\n")

    # 5. Optional EFA
    if opts.include_efa:
        L("# Exploratory Factor Analysis (parallel analysis)")
        all_items = [i for s in scales for i in s.items]
        all_items_vec = ", ".join(f"'{i}'" for i in all_items)
        L(f"efa_items <- na.omit(data[, c({all_items_vec})])")
        L("psych::fa.parallel(efa_items, fa = 'fa')")
        L("# Set nfactors manually based on the scree / parallel output")
        L("fa_res <- psych::fa(efa_items, nfactors =   , rotate = 'promax')\n")

    # 6. Descriptives & correlations
    L("# Descriptive statistics and correlations")
    L("desclist <- purrr::map_df(names(data), ~psych::describe(data[[.x]]), .id = 'variable')")
    cor_expr = {
        "pearson": "cor(data, use = 'pairwise.complete.obs')",
        "spearman": "cor(data, method = 'spearman', use = 'pairwise.complete.obs')",
        "polychoric": "psych::polychoric(data)$rho",
    }[opts.correlation_type]
    L(f"cors <- {cor_expr}\n")

    # 7. Save artefacts
    L("haven::write_sav(data, 'enhanced_dataset.sav')")
    L("save(desclist, cors, file = 'analysis_objects.RData')\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Write R file to disk
# ---------------------------------------------------------------------------

def write_r_file(r_script: str, out_dir: str | os.PathLike = "outputs") -> str:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    r_path = pathlib.Path(out_dir) / "analysis.R"
    r_path.write_text(r_script, encoding="utf-8")
    return str(r_path)


# ---------------------------------------------------------------------------
# 9. Session logging to JSON Lines
# ---------------------------------------------------------------------------

LOG_FILE = pathlib.Path("session_log.jsonl")


def log_session(event: Dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        json.dump(event, fp, default=str)
        fp.write("\n")


# ---------------------------------------------------------------------------
# 10. Orchestrator (batch)
# ---------------------------------------------------------------------------

def orchestrate_pipeline(path: str | bytes | BytesIO, opts: Options) -> str:
    df, meta = load_sav(path)
    clean_df, clean_meta = sanitize_metadata(df, meta)
    var_view = summarize_variables(clean_meta)
    scales_prop = gemini_detect_scales(var_view, PromptConfig(system_prompt="", temperature=0.2, top_p=0.9))
    rev_map = detect_reverse_items(scales_prop, clean_df)

    sav_name_for_r = meta.get("sav_name", "data.sav")
    r_script = build_r_syntax(sav_path=sav_name_for_r, scales=scales_prop, rev_map=rev_map, opts=opts)
    out_path = write_r_file(r_script)

    log_session({
        "timestamp": datetime.now().isoformat(),
        "file": sav_name_for_r,
        "n_scales": len(scales_prop),
        "opts": asdict(opts),
    })

    return out_path
