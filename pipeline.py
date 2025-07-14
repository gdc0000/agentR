from __future__ import annotations

"""Core pipeline utilities for the Survey‑to‑R Agent.

Each function conforms to the interfaces outlined during design so that the
Streamlit GUI (``streamlit_app.py``) can import them directly.

NOTE
----
• Implementations are *minimal viable* and focus on I/O integrity. Replace the
  TODO sections with domain‑specific logic or API calls as needed.
• No heavy imports inside functions that run on every call—expensive / optional
  libraries are imported lazily.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Literal, Tuple
import json
import logging
import os
import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Data‑classes
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
    # dataclasses
    "VariableInfo",
    "Scale",
    "PromptConfig",
    "Options",
    # pipeline functions
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
# Utility helpers
# ---------------------------------------------------------------------------

def _bytesio_to_pathlike(path_or_bytes) -> pathlib.Path | BytesIO:
    """Accept str / Path / UploadedFile / Bytes and return bytes‑compatible obj."""
    if isinstance(path_or_bytes, (str, os.PathLike)):
        return pathlib.Path(path_or_bytes)
    return path_or_bytes  # e.g. BytesIO uploaded file


# ---------------------------------------------------------------------------
# 1. Load .sav
# ---------------------------------------------------------------------------

def load_sav(path: str | BytesIO) -> Tuple[pd.DataFrame, Dict]:
    """Read SPSS .sav file using ``pyreadstat``.

    Parameters
    ----------
    path
        Path to .sav file or BytesIO‐like object.

    Returns
    -------
    (df, meta)
        df  : DataFrame containing raw data.
        meta: Dict with variable labels, value labels, types, missing codes.
    """
    # Lazy import to avoid pyreadstat requirement at startup
    import pyreadstat

    path_obj = _bytesio_to_pathlike(path)

    df, meta = pyreadstat.read_sav(path_obj, apply_value_formats=False)
    meta_dict = {
        "var_labels": meta.column_labels,
        "value_labels": meta.variable_value_labels,
        "var_types": meta.original_variable_types,
        "missing_ranges": meta.missing_ranges,  # dict var -> list[tuple]
    }
    return df, meta_dict


# ---------------------------------------------------------------------------
# 2. Sanitize metadata
# ---------------------------------------------------------------------------

def sanitize_metadata(
    df: pd.DataFrame, meta: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Basic cleaning: drop obvious non‑item columns, normalise missings.

    Very conservative: only removes string columns exceeding 50 unique values or
    names like *id*, *timestamp*, *date*.
    """
    drop_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" and df[c].nunique() > 50
        or c.lower() in {"id", "participant_id", "timestamp", "date"}
    ]
    clean_df = df.drop(columns=drop_cols)

    # Replace user‑defined missing codes (e.g., 999 or 99) with NaN
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
    from statistics import mean

    var_labels = clean_meta.get("var_labels", {})
    # missing pct must be calculated later when df is given; here set 0 placeholder
    var_view: List[VariableInfo] = []
    for name, label in var_labels.items():
        var_view.append(
            VariableInfo(
                name=name,
                label=label,
                item_text=None,
                missing_pct=0.0,
                type="numeric",
            )
        )
    return var_view


# ---------------------------------------------------------------------------
# 4. LLM: detect scales (Gemini stub)
# ---------------------------------------------------------------------------

def gemini_detect_scales(
    var_view: List[VariableInfo], prompt_cfg: PromptConfig
) -> List[Scale]:
    """Call Gemini API to get constructs suggestions.

    The current implementation is a stub that groups items by prefix before the
    first underscore (e.g., *anx_1, anx_2* → *anx*).
    """

    clusters: Dict[str, List[str]] = {}
    for v in var_view:
        prefix = v.name.split("_")[0] if "_" in v.name else "misc"
        clusters.setdefault(prefix, []).append(v.name)

    scales: List[Scale] = []
    for pref, items in clusters.items():
        scales.append(Scale(name=pref.title(), items=items, confidence=0.3))
    return scales


# ---------------------------------------------------------------------------
# 6. Detect reverse items
# ---------------------------------------------------------------------------

def detect_reverse_items(
    scales_confirmed: List[Scale], df: pd.DataFrame
) -> Dict[str, bool]:
    """Very naive heuristic: if an item correlates negatively with the total of its scale it is flagged as reversed."""
    rev_map: Dict[str, bool] = {}
    for scale in scales_confirmed:
        items = scale.items
        if len(items) < 2:
            continue
        sub = df[items].dropna()
        if sub.empty:
            continue
        scale_score = sub.mean(axis=1)
        for item in items:
            corr = sub[item].corr(scale_score)
            rev_map[item] = True if corr < 0 else False
    return rev_map


# ---------------------------------------------------------------------------
# 7. Build R syntax
# ---------------------------------------------------------------------------

def build_r_syntax(
    sav_path: str,
    scales: List[Scale],
    rev_map: Dict[str, bool],
    opts: Options,
) -> str:
    """Create an R script as a string based on the confirmed scales and options."""

    def indent(text: str, spaces: int = 4) -> str:
        pad = " " * spaces
        return "\n".join(pad + line if line.strip() else line for line in text.splitlines())

    lines: List[str] = [
        "# ------------------------------------------------------------------",
        "# Auto‑generated R syntax (Survey‑to‑R Agent)",
        f"# Generated: {datetime.now().isoformat()}",
        "# ------------------------------------------------------------------",
        "\n# 1. Load packages",
        "if (!require('pacman')) install.packages('pacman')",
        "pacman::p_load(haven, tidyverse, psych, MBESS, GPArotation, lavaan)",
        "\n# 2. Import data",
        f"data <- haven::read_sav('{sav_path}')",
    ]

    # 3. Reverse items
    if any(rev_map.values()):
        lines.append("\n# 3. Reverse‑score items")
        for item, to_reverse in rev_map.items():
            if to_reverse:
                lines.append(f"data${item} <- max(data${item}, na.rm = TRUE) + min(data${item}, na.rm = TRUE) - data${item}")

    # 4. Reliability per scale
    lines.append("\n# 4. Reliability analysis")
    for scale in scales:
        vec_name = f"items_{scale.name.lower()}"
        items_vec = ", ".join(f"'{i}'" for i in scale.items)
        lines.append(f"{vec_name} <- c({items_vec})")
        lines.append(
            f"psych::alpha(data[{vec_name}], check.keys = TRUE, na.rm = TRUE)$total"  # Cronbach α
        )
        lines.append(
            f"MBESS::ci.reliability(data[{vec_name}], type = 'omega')  # McDonald Ω CI"
        )
    # 5. (Optional) EFA
    if opts.include_efa:
        lines.append("\n# 5. Exploratory Factor Analysis (parallel analysis)")
        all_items = [i for s in scales for i in s.items]
        all_items_vec = ", ".join(f"'{i}'" for i in all_items)
        lines.extend(
            [
                f"fa_items <- data[, c({all_items_vec})]",
                "fa_items <- na.omit(fa_items)",
                "psych::fa.parallel(fa_items, fa = 'fa')",
                "fa_result <- psych::fa(fa_items, nfactors =  , rotate = 'promax')  # <-- set nfactors manually based on parallel",
                "fa_result$loadings",
            ]
        )

    # 6. Descriptives & Correlations
    lines.append("\n# 6. Descriptive stats and correlations")
    lines.append("desclist <- purrr::map_df(names(data), ~psych::describe(data[[.x]]), .id = 'variable')")
    cor_fun = {
        "pearson": "cor",
        "spearman": "cor(method = 'spearman')",
        "polychoric": "psych::polychoric"
    }[opts.correlation_type]
    lines.append(f"cors <- {cor_fun}(data, use = '{opts.missing_strategy}')")

    # 7. Save outputs
    lines.append("\n# 7. Save enhanced dataset and analysis objects")
    lines.append("haven::write_sav(data, 'enhanced_dataset.sav')")
    lines.append("save(desclist, cors, file = 'analysis_objects.RData')")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Write R file
# ---------------------------------------------------------------------------

def write_r_file(r_script: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = pathlib.Path(out_dir) / "analysis.R"
    path.write_text(r_script, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 9. Session logging
# ---------------------------------------------------------------------------

LOG_PATH = pathlib.Path("session_log.jsonl")


def log_session(event: Dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        json.dump(event, fp, default=str)
        fp.write("\n")


# ---------------------------------------------------------------------------
# 10. Orchestrator (batch, non‑GUI)
# ---------------------------------------------------------------------------

def orchestrate_pipeline(path: str | BytesIO, opts: Options) -> str:
    df, meta = load_sav(path)
    clean_df, clean_meta = sanitize_metadata(df, meta)
    var_view = summarize_variables(clean_meta)

    prompt_cfg = PromptConfig(system_prompt="Group survey items", temperature=0.2, top_p=0.9)
    scales_prop = gemini_detect_scales(var_view, prompt_cfg)
    # NOTE: In a non‑interactive context we assume scales_prop is accepted as‑is.
    rev_map = detect_reverse_items(scales_prop, clean_df)

    r_script = build_r_syntax(
        sav_path=str(path) if isinstance(path, (str, os.PathLike)) else "uploaded_file.sav",
        scales=scales_prop,
        rev_map=rev_map,
        opts=opts,
    )

    out_path = write_r_file(r_script, out_dir="outputs")

    log_session(
        {
            "timestamp": datetime.now().isoformat(),
            "file": str(path),
            "n_scales": len(scales_prop),
            "opts": asdict(opts),
        }
    )
    return out_path
