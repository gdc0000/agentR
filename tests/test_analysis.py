"""
Unit tests for the survey_to_r.analysis module.
"""

import pytest
from survey_to_r.analysis import summarize_variables, detect_reverse_items
from survey_to_r.models import VariableInfo


def test_summarize_variables_empty_metadata():
    """Test summarize_variables with empty metadata."""
    clean_meta = {
        "var_labels": {},
        "dropped": []
    }
    
    result = summarize_variables(clean_meta)
    assert result == []


def test_summarize_variables_basic():
    """Test summarize_variables with basic metadata."""
    clean_meta = {
        "var_labels": {"var1": "Variable 1", "var2": "Variable 2"},
        "dropped": []
    }
    
    result = summarize_variables(clean_meta)
    assert len(result) == 2
    
    # Check first variable
    var1 = next(v for v in result if v.name == "var1")
    assert var1.name == "var1"
    assert var1.label == "Variable 1"
    assert var1.item_text is None
    assert var1.missing_pct == 0.0
    assert var1.type == "numeric"
    
    # Check second variable
    var2 = next(v for v in result if v.name == "var2")
    assert var2.name == "var2"
    assert var2.label == "Variable 2"
    assert var2.item_text is None
    assert var2.missing_pct == 0.0
    assert var2.type == "numeric"


def test_summarize_variables_with_missing_data():
    """Test summarize_variables with missing data in some fields."""
    clean_meta = {
        "var_labels": {"var1": "Variable 1", "var2": None},
        "dropped": []
    }
    
    result = summarize_variables(clean_meta)
    assert len(result) == 2
    
    var1 = next(v for v in result if v.name == "var1")
    assert var1.label == "Variable 1"
    
    var2 = next(v for v in result if v.name == "var2")
    assert var2.label is None  # None value in var_labels
    assert var2.item_text is None


def test_detect_reverse_items_no_scales():
    """Test detect_reverse_items with empty scales list."""
    import pandas as pd
    df = pd.DataFrame()
    scales = []
    reverse_map = detect_reverse_items(scales, df)
    assert reverse_map == {}


def test_detect_reverse_items_basic():
    """Test detect_reverse_items with scales containing reverse items."""
    import pandas as pd
    from survey_to_r.models import Scale
    
    # Create test data with reverse correlation
    df = pd.DataFrame({
        'item1': [1, 2, 3, 4, 5],
        'item2': [5, 4, 3, 2, 1],  # Reverse correlated
        'item3': [1, 2, 3, 4, 5]
    })
    
    scales = [Scale(name="test_scale", items=["item1", "item2", "item3"])]
    
    reverse_map = detect_reverse_items(scales, df)
    
    # item2 should be detected as reverse-scored due to negative correlation
    # item1 and item3 should be positively correlated with scale score
    assert reverse_map == {"item1": False, "item2": True, "item3": False}


def test_detect_reverse_items_single_item_scale():
    """Test detect_reverse_items with scale containing only one item."""
    import pandas as pd
    from survey_to_r.models import Scale
    
    df = pd.DataFrame({
        'item1': [1, 2, 3, 4, 5]
    })
    
    scales = [Scale(name="single_item_scale", items=["item1"])]
    
    reverse_map = detect_reverse_items(scales, df)
    
    # Single item scales should be skipped
    assert reverse_map == {}


def test_detect_reverse_items_empty_dataframe():
    """Test detect_reverse_items with empty DataFrame."""
    import pandas as pd
    from survey_to_r.models import Scale
    
    df = pd.DataFrame()
    scales = [Scale(name="test_scale", items=["item1", "item2"])]
    
    reverse_map = detect_reverse_items(scales, df)
    
    # Empty DataFrame should result in empty reverse map
    assert reverse_map == {}


def test_detect_reverse_items_non_numeric_data():
    """Test detect_reverse_items with non-numeric data."""
    import pandas as pd
    from survey_to_r.models import Scale
    
    df = pd.DataFrame({
        'item1': [1, 2, 3, 4, 5],
        'item2': ['a', 'b', 'c', 'd', 'e']  # Non-numeric
    })
    
    scales = [Scale(name="test_scale", items=["item1", "item2"])]
    
    reverse_map = detect_reverse_items(scales, df)
    
    # Only numeric items should be processed
    assert reverse_map == {}