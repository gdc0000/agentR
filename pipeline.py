from __future__ import annotations

"""Core pipeline utilities for the Survey‑to‑R Agent – **rev 2** (2025‑07‑14)

Fixes
-----
* `load_sav` now stores **`var_labels` as a *dict*** (`name → label`) using
  ``meta.column_names_to_labels`` (instead of the previous list), plus a new
  key `var_names` for column order.  
  → This resolves ``AttributeError: 'list' object has no attribute 'items'`` in
  `summarize_variables`.
* `summarize_variables` still accepts fallback formats (list) but expects a
  dict in normal flow.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Literal, Tuple
import json
import os
import pathlib
import tempfile
import uuid

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
    """Read an SPSS **.sav** file from path, bytes or file‑like (Streamlit `UploadedFile`)."""
    import pyreadstat

    # Determine type and get bytes or path
    if isinstance(path_or_file, (str, os.PathLike)):
        sav_path = os.fspath(path_or_file)
        df, meta = pyreadstat.read_sav(sav_path, apply_value_formats=False)
    else:
        # Bytes or file‑like → write to temp file
        if isinstance(path_or_file, (bytes, bytearray)):
            raw_bytes: bytes = bytes(path_or_file)
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

    meta_dict = {
        "sav_name": sav_path,
        "var_labels": meta.column_names_to_labels,  # dict name → label
        "var_names": meta.column_names,             # ordered names list
        "value_labels": meta.variable_value_labels,
        "var_types": meta.original_variable_types,
        "missing_ranges": meta.missing_ranges,
    }
    return df, meta_dict


# ---------------------------------------------------------------------------
# 2. Sanitize metadata (placeholder)
# ---------------------------------------------------------------------------

def sanitize_metadata(df: pd.DataFrame, meta: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Basic cleaning: drop obvious non‑scale columns and normalise missings."""
    drop_cols: List[str] = []
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() > 50:
            drop_cols.append(col)
        elif col.lower() in {"id", "participant_id", "timestamp", "date"}:
            drop_cols.append(col)

    clean_df = df.drop(columns=drop_cols)

    for col, ranges in meta.get("missing_ranges", {}).items():
        if col not in clean_df.columns:
            continue
        for lo, hi in ranges:
            clean_df.loc[clean_df[col].between(lo, hi), col] = pd.NA

    clean_meta = meta.copy()
    clean_meta["dropped"] = drop_cols
    return clean_df, clean_meta


# ---------------------------------------------------------------------------
# 3. Summarize variables
# ---------------------------------------------------------------------------

def summarize_variables(clean_meta: Dict) -> List[VariableInfo]:
    """Return a list of `VariableInfo` objects for the *variable view* table."""
    var_labels = clean_meta.get("var_labels", {})

    # Ensure dict mapping; fallback if list supplied
    if isinstance(var_labels, list):
        names = clean_meta.get("var_names", list(range(len(var_labels))))
        var_labels = {n: l for n, l in zip(names, var_labels)}

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
# 4. Gemini detect scales (dummy implementation)
# ---------------------------------------------------------------------------

def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    clusters: Dict[str, List[str]] = {}
    for v in var_view:
        pref = v.name.split("_")[0] if "_" in v.name else "misc"
        clusters.setdefault(pref, []).append(v.name)
    return [Scale(name=k.title(), items=items, confidence=0.3) for k, items in clusters.items()]


# ---------------------------------------------------------------------------
# 6. Detect reverse items (simple corr sign)
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
# 7. Build R syntax (unchanged)
# ---------------------------------------------------------------------------

def build_r_syntax(sav_path: str, scales: List[Scale], rev_map: Dict[str, bool], opts: Options) -> str:
    lines: List[str] = []
    L = lines.append

    L("# -------------------------------------------------------------")
    L("# Auto‑generated R syntax (Survey‑to‑R Agent)")
    L(f"# Generated: {datetime.now().isoformat()}")
    L("# -------------------------------------------------------------\n")

    L("if (!require('pacman')) install.packages('pacman')")
    L("pacman::p_load(haven, tidyverse, psych, MBESS, GPArotation, lavaan)")

    L(f"data <- haven::read_sav('{sav_path}')\n")

    if any(rev_map.values()):
        L("# Reverse‑score flagged items")
        for item, flag in rev_map.items():
            if flag:
                L(f"maxv <- max(data${item}, na.rm = TRUE)")
                L(f"minv <- min(data${item}, na.rm = TRUE)")
                L(f"data${item} <- maxv + minv - data${item}\n")

    L("# Reliability analysis (Cronbach α / McDonald Ω)")
    for sc in scales:
        vname = f"items_{sc.name.lower()}"
        items_csv = ", ".join(f"'{i}'" for i in sc.items)
        L(f"{vname} <- c({items_csv})")
        L(f"alpha_{sc.name.lower()} <- psych::alpha(data[{vname}], check.keys = TRUE)")
        L(f"omega_{sc.name.lower()} <- MBESS::ci.reliability(data[{vname}], type = 'omega')\n")

    if opts.include_efa:
        L("# Exploratory Factor Analysis (parallel)")
        all_items = [i for s in scales for i in s.items]
        all_items_vec = ", ".join(f"'{i}'" for i in all_items)
        L(f"efa_items <- na.omit(data[, c({all_items_vec})])")
        L("psych::fa.parallel(efa_items, fa = 'fa')")
        L("fa_res <- psych::fa(efa_items, nfactors =   , rotate = 'promax')\n")

    L("# Descriptive statistics and correlations")
    L("desclist <- purrr::map_df(names(data), ~psych::describe(data[[.x]]), .id = 'variable')")
    cor_expr = {
        "pearson": "cor(data, use = 'pairwise.complete.obs')",
        "spearman": "cor(data, method = 'spearman', use = 'pairwise.complete.obs')",
        "polychoric": "psych::polychoric(data)$rho",
    }[opts.correlation_type]
    L(f"cors <- {cor_expr}\n")

    L("haven::write_sav(data, 'enhanced_dataset.sav')")
    L("save(desclist, cors, file = 'analysis_objects.RData')\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Write R file
# ---------------------------------------------------------------------------

def write_r_file(r_script: str, out_dir: str | os.PathLike = "outputs") -> str:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    r_path = pathlib.Path(out_dir) / "analysis.R"
    r_path.write_text(r_script, encoding="utf-8")
    return str(r_path)


# ---------------------------------------------------------------------------
# 9. Log session
# ---------------------------------------------------------------------------

LOG_FILE = pathlib.Path("session_log.jsonl")

def log_session(event: Dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        json.dump(event, fp, default=str)
        fp.write("\n")


# ---------------------------------------------------------------------------
# 10. Orchestrator
# ---------------------------------------------------------------------------

def orchestrate_pipeline(path: str | bytes | BytesIO, opts: Options) -> str:
    df, meta = load_sav(path)
    clean_df, clean_meta = sanitize_metadata(df, meta)
    var_view = summarize_variables(clean_meta)
    scales = gemini_detect_scales(var_view, PromptConfig(system_prompt="", temperature=0.2, top_p=0.9))
    rev_map = detect_reverse_items(scales, clean_df)

    r_script = build_r_syntax(meta["sav_name"], scales, rev_map, opts)
    out_path = write_r_file(r_script)

    log
