"""
R syntax generation for the Survey-to-R Agent.

This module handles the generation of R analysis scripts based on
detected scales and analysis options.
"""

from datetime import datetime
from typing import Dict, List

from .models import Scale, Options


def build_r_syntax(sav_path: str, scales: List[Scale], rev_map: Dict[str, bool], opts: Options) -> str:
    """
    Build R syntax for statistical analysis.
    
    Args:
        sav_path: Path to the SPSS file
        scales: List of confirmed scales
        rev_map: Dictionary mapping items to reverse-scoring flags
        opts: Analysis options
        
    Returns:
        Complete R script as a string
    """
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