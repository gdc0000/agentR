from __future__ import annotations

"""pipeline.py – Core utilities for the Survey‑to‑R Agent  
rev 4 (2025‑07‑14)

Fixes
-----
* **Streamlit multiselect default error**: `summarize_variables` now filters out
  variables that were dropped during cleaning, ensuring that `scale.items` is
  always a subset of the selectable *options*.
* `sanitize_metadata` prunes dropped variables from **all** metadata dicts so
  subsequent steps no longer reference removed columns.
* Lazy‑import strategy (rev 3) retained.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Literal, Tuple
import json
import os
import pathlib
import tempfile
import uuid

# ---------------------------------------------------------------------------
# Optional heavy dependencies – loaded lazily
# ---------------------------------------------------------------------------

try:
    import pandas as _pd  # noqa: F401 – only to test availability
    _PANDAS_AVAILABLE = True
except ModuleNotFoundError as _e:
    _PANDAS_AVAILABLE = False if _e.name == "micropip" else True

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
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pandas():  # -> pandas
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required but not available in this environment.")
    import pandas as pd
    return pd

# ---------------------------------------------------------------------------
# 1. Load .sav
# ---------------------------------------------------------------------------

def load_sav(path_or_file) -> Tuple[Any, Dict]:
    pd = _ensure_pandas()
    try:
        import pyreadstat  # heavy; local import
    except ModuleNotFoundError:
        raise RuntimeError("pyreadstat is required to read .sav files but is not installed.")

    if isinstance(path_or_file, (str, os.PathLike)):
        sav_path = os.fspath(path_or_file)
        df, meta = pyreadstat.read_sav(sav_path, apply_value_formats=False)
    else:
        # Bytes or file‑like → temp file
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

# ---------------------------------------------------------------------------
# 2. Sanitize metadata
# ---------------------------------------------------------------------------

def sanitize_metadata(df: Any, meta: Dict) -> Tuple[Any, Dict]:
    pd = _ensure_pandas()

    drop_cols: List[str] = []
    for col in df.columns:
        if (df[col].dtype == "object" and df[col].nunique() > 50) or col.lower() in {"id", "participant_id", "timestamp", "date"}:
            drop_cols.append(col)

    clean_df = df.drop(columns=drop_cols)

    for col, ranges in meta.get("missing_ranges", {}).items():
        if col not in clean_df.columns:
            continue
        for lo, hi in ranges:
            clean_df.loc[clean_df[col].between(lo, hi), col] = pd.NA

    # -------- prune metadata -------- #
    clean_meta = meta.copy()
    clean_meta["dropped"] = drop_cols
    for key in ("var_labels", "var_types", "missing_ranges"):
        if key in clean_meta and isinstance(clean_meta[key], dict):
            clean_meta[key] = {k: v for k, v in clean_meta[key].items() if k not in drop_cols}
    clean_meta["var_names"] = [n for n in meta.get("var_names", []) if n not in drop_cols]

    return clean_df, clean_meta

# ---------------------------------------------------------------------------
# 3. Summarize variables (post‑clean)
# ---------------------------------------------------------------------------

def summarize_variables(clean_meta: Dict) -> List[VariableInfo]:
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

# ---------------------------------------------------------------------------
# 4. Gemini detect scales (dummy impl.)
# ---------------------------------------------------------------------------

def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    clusters: Dict[str, List[str]] = {}
    for v in var_view:
        prefix = v.name.split("_")[0] if "_" in v.name else "misc"
        clusters.setdefault(prefix, []).append(v.name)
    return [Scale(name=k.title(), items=items, confidence=0.3) for k, items in clusters.items()]

# ---------------------------------------------------------------------------
# 5. Detect reverse items
# ---------------------------------------------------------------------------

def detect_reverse_items(scales_confirmed: List[Scale], df: Any) -> Dict[str, bool]:
    pd = _ensure_pandas()
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
# 6. Build R syntax
# ---------------------------------------------------------------------------

def build_r_syntax(sav_path: str, scales: List[Scale], rev_map: Dict[str, bool], opts: Options) -> str:
    lines: List[str] = []
    a = lines.append

    a("# -------------------------------------------------------------")
    a("# Auto‑generated R syntax (Survey‑to‑R Agent)")
    a(f"# Generated: {datetime.now().isoformat()}")
    a("# -------------------------------------------------------------\n")
    a("if (!require('pacman')) install.packages('pacman')")
    a("pacman::p_load(haven, tidyverse, psych, MBESS, GPArotation, lavaan)")
    a(f"data <- haven::read_sav('{sav_path}')\n")

    if any(rev_map.values()):
        a("# Reverse‑score flagged items")
        for item, flag in rev_map.items():
            if flag:
                a(f"maxv <- max(data${item}, na.rm = TRUE)")
                a(f"minv <- min(data${item}, na.rm = TRUE)")
                a(f"data${item} <- maxv + minv - data${item}\n")

    a("# Reliability analysis (Cronbach α / McDonald Ω)")
    for sc in scales:
        vname = f"items_{sc.name.lower()}"
        items_csv = ", ".join(f"'{i}'" for i in sc.items)
        a(f"{vname} <- c({items_csv})")
        a(f"alpha_{sc.name.lower()} <- psych::alpha(data[{vname}], check.keys = TRUE)")
        a(f"omega_{sc.name.lower()} <- MBESS::ci.reliability(data[{vname}], type = 'omega')\n")

    if opts.include_efa:
        a("# Exploratory Factor Analysis (parallel)")
        all_items = [i for s in scales for i in s.items]
        items_vec = ", ".join(f"'{i}'" for i in all_items)
        a(f"efa_items <- na.omit(data[, c({items_vec})])")
        a("psych::fa.parallel(efa_items, fa = 'fa')")
        a("fa_res <- psych::fa(efa_items, nfactors =   , rotate = 'promax')\n")

    a("# Descriptive statistics and correlations")
    a("desclist <- purrr::map_df(names(data), ~psych::describe(data[[.x]]), .id = 'variable')")
    cor_expr = {
        "pearson": "cor(data, use = 'pairwise.complete.obs')",
        "spearman": "cor(data, method = 'spearman', use = 'pairwise.complete.obs')",
        "polychoric": "psych::polychoric(data)$rho",
    }[opts.correlation_type]
    a(f"cors <- {cor_expr}\n")

    a("haven::write_sav(data, 'enhanced_dataset.sav')")
    a("save(desclist, cors, file = 'analysis_objects.RData')\n")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 7. Write R file
# ---------------------------------------------------------------------------

def write_r_file(r_script: str, out_dir: str | os.PathLike = "outputs") -> str:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    r_path = pathlib.Path(out_dir) / "analysis.R"
    r_path.write_text(r_script, encoding="utf-8")
    return str(r_path)

# ---------------------------------------------------------------------------
# 8. Logging
# ---------------------------------------------------------------------------

LOG_FILE = pathlib.Path("session_log.json
