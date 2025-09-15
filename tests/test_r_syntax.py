"""
Unit tests for the survey_to_r.r_syntax module.
"""

import pytest
from survey_to_r.r_syntax import build_r_syntax
from survey_to_r.models import Scale, Options


def test_build_r_syntax_no_scales():
    """Test build_r_syntax with no scales provided."""
    sav_path = "test_data.sav"
    scales = []
    rev_map = {}
    opts = Options()
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should contain basic R code for loading data but no scale analysis
    assert "library(haven)" in result
    assert "test_data.sav" in result
    assert "scale" not in result.lower()  # No scale analysis


def test_build_r_syntax_basic_scales():
    """Test build_r_syntax with basic scales."""
    sav_path = "survey.sav"
    scales = [
        Scale(name="Satisfaction", items=["q1", "q2", "q3"], confidence=0.8),
        Scale(name="Engagement", items=["q4", "q5"], confidence=0.7),
    ]
    rev_map = {"q2": True, "q5": True}
    opts = Options()
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should contain scale definitions and reverse coding
    assert "Satisfaction" in result
    assert "Engagement" in result
    assert "q2 = 6 - q2" in result  # Reverse coding
    assert "q5 = 6 - q5" in result  # Reverse coding
    assert "alpha(Satisfaction)" in result  # Reliability analysis


def test_build_r_syntax_with_efa():
    """Test build_r_syntax with EFA enabled."""
    sav_path = "data.sav"
    scales = [
        Scale(name="Scale1", items=["item1", "item2"], confidence=0.9),
    ]
    rev_map = {}
    opts = Options(include_efa=True)
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should contain EFA code
    assert "factanal" in result
    assert "efa_results" in result


def test_build_r_syntax_without_efa():
    """Test build_r_syntax with EFA disabled."""
    sav_path = "data.sav"
    scales = [
        Scale(name="Scale1", items=["item1", "item2"], confidence=0.9),
    ]
    rev_map = {}
    opts = Options(include_efa=False)
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should not contain EFA code
    assert "factanal" not in result
    assert "efa_results" not in result


def test_build_r_syntax_correlation_types():
    """Test build_r_syntax with different correlation types."""
    sav_path = "data.sav"
    scales = [
        Scale(name="Scale1", items=["item1", "item2"], confidence=0.9),
    ]
    rev_map = {}
    
    # Test pearson correlation
    opts_pearson = Options(correlation_type="pearson")
    result_pearson = build_r_syntax(sav_path, scales, rev_map, opts_pearson)
    assert "cor(Scale1, method = \"pearson\")" in result_pearson
    
    # Test spearman correlation
    opts_spearman = Options(correlation_type="spearman")
    result_spearman = build_r_syntax(sav_path, scales, rev_map, opts_spearman)
    assert "cor(Scale1, method = \"spearman\")" in result_spearman
    
    # Test kendall correlation
    opts_kendall = Options(correlation_type="kendall")
    result_kendall = build_r_syntax(sav_path, scales, rev_map, opts_kendall)
    assert "cor(Scale1, method = \"kendall\")" in result_kendall


def test_build_r_syntax_missing_strategies():
    """Test build_r_syntax with different missing value strategies."""
    sav_path = "data.sav"
    scales = [
        Scale(name="Scale1", items=["item1", "item2"], confidence=0.9),
    ]
    rev_map = {}
    
    # Test listwise deletion
    opts_listwise = Options(missing_strategy="listwise")
    result_listwise = build_r_syntax(sav_path, scales, rev_map, opts_listwise)
    assert "na.omit" in result_listwise
    
    # Test pairwise deletion
    opts_pairwise = Options(missing_strategy="pairwise")
    result_pairwise = build_r_syntax(sav_path, scales, rev_map, opts_pairwise)
    assert "use = \"pairwise.complete.obs\"" in result_pairwise


def test_build_r_syntax_with_scale_notes():
    """Test build_r_syntax includes scale notes when present."""
    sav_path = "data.sav"
    scales = [
        Scale(
            name="ScaleWithNote", 
            items=["item1", "item2"], 
            confidence=0.9,
            note="This scale measures satisfaction"
        ),
    ]
    rev_map = {}
    opts = Options()
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should include the note as a comment
    assert "# ScaleWithNote: This scale measures satisfaction" in result


def test_build_r_syntax_reverse_coding_logic():
    """Test build_r_syntax handles reverse coding correctly."""
    sav_path = "data.sav"
    scales = [
        Scale(name="TestScale", items=["q1", "q2_r", "q3"], confidence=0.8),
    ]
    rev_map = {"q2_r": True}
    opts = Options()
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should reverse code only the items in rev_map
    assert "q2_r = 6 - q2_r" in result
    assert "q1 = 6 - q1" not in result  # Not in rev_map
    assert "q3 = 6 - q3" not in result  # Not in rev_map


def test_build_r_syntax_data_loading():
    """Test build_r_syntax includes proper data loading code."""
    sav_path = "/path/to/data.sav"
    scales = []
    rev_map = {}
    opts = Options()
    
    result = build_r_syntax(sav_path, scales, rev_map, opts)
    
    # Should include correct path handling
    assert "read_sav(\"/path/to/data.sav\")" in result
    assert "library(haven)" in result
    assert "data <-" in result