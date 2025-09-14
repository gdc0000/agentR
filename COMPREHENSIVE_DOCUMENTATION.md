# Survey-to-R Agent: Comprehensive Documentation

## Table of Contents
1. [Overview and Purpose](#overview-and-purpose)
2. [Architecture Overview](#architecture-overview)
3. [Detailed Workflow](#detailed-workflow)
4. [Code Structure Analysis](#code-structure-analysis)
5. [Data Classes and Models](#data-classes-and-models)
6. [Key Functions Deep Dive](#key-functions-deep-dive)
7. [User Interface Flow](#user-interface-flow)
8. [Current Limitations](#current-limitations)
9. [Improvement Recommendations](#improvement-recommendations)
10. [Technical Requirements](#technical-requirements)

## Overview and Purpose

The **Survey-to-R Agent** is an intelligent web application designed to streamline the transition from SPSS survey data to R statistical analysis. This tool addresses a common pain point in psychological and social science research where data collection often happens in SPSS but analysis is performed in R.

### Primary Use Cases
- **Psychology Researchers**: Automatically identify psychological constructs and scales from survey data
- **Social Scientists**: Convert SPSS datasets to R with intelligent variable grouping
- **Data Analysts**: Generate publication-ready R analysis scripts with minimal manual coding
- **Survey Researchers**: Detect reverse-scored items and construct reliability measures automatically

### Key Value Propositions
1. **AI-Powered Construct Detection**: Uses language models to intelligently group survey items into psychological constructs
2. **Automated Reverse Item Detection**: Identifies items that should be reverse-scored based on correlation patterns
3. **Ready-to-Use R Scripts**: Generates comprehensive R syntax including reliability analysis, factor analysis, and descriptive statistics
4. **Interactive Review Process**: Allows researchers to validate and modify AI suggestions before final script generation
5. **Metadata Preservation**: Maintains SPSS variable labels and handles missing data appropriately

## Architecture Overview

The application follows a clean separation of concerns with two main components:

```
┌─────────────────┐    ┌──────────────────────┐
│   main.py       │    │    pipeline.py       │
│                 │    │                      │
│ • Streamlit UI  │◄──►│ • Core Logic         │
│ • User Input    │    │ • Data Processing    │
│ • State Mgmt    │    │ • AI Integration     │
│ • File Handling │    │ • R Script Generation│
└─────────────────┘    └──────────────────────┘
```

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: pandas, pyreadstat (SPSS file handling)
- **Statistical Analysis**: Built-in correlation and statistical functions
- **AI Integration**: Google Gemini API (currently with dummy implementation)
- **Output**: R syntax generation with statistical packages

## Detailed Workflow

### Phase 1: Data Ingestion and Cleaning
```
User uploads .sav file
        ↓
load_sav() reads SPSS file with metadata
        ↓
sanitize_metadata() cleans and filters variables
        ↓
summarize_variables() creates variable overview
```

### Phase 2: AI-Powered Construct Detection
```
Variable metadata sent to Gemini AI
        ↓
gemini_detect_scales() groups items into constructs
        ↓
Results cached in Streamlit session state
```

### Phase 3: Human Review and Validation
```
AI proposals displayed in expandable UI
        ↓
User can modify construct names and item assignments
        ↓
User confirms final construct selection
```

### Phase 4: Statistical Analysis Preparation
```
detect_reverse_items() analyzes item correlations
        ↓
Identifies items needing reverse scoring
        ↓
Prepares statistical analysis options
```

### Phase 5: R Script Generation
```
build_r_syntax() creates comprehensive R script
        ↓
Includes: data loading, reverse scoring, reliability analysis, EFA, descriptives
        ↓
User downloads ready-to-run R script
```

## Code Structure Analysis

### main.py Structure
```python
# Configuration and Setup
st.set_page_config()

# Main Application Logic
def main():
    # 1. UI Setup and Sidebar Options
    # 2. File Upload Handler  
    # 3. Data Loading and Processing
    # 4. AI Construct Detection (with caching)
    # 5. Interactive Construct Review
    # 6. R Script Generation and Download
```

### pipeline.py Structure
```python
# Data Models and Configuration
@dataclass classes for structured data

# Core Pipeline Functions
1. load_sav()           # SPSS file ingestion
2. sanitize_metadata()  # Data cleaning
3. summarize_variables()# Variable analysis
4. gemini_detect_scales()# AI construct detection
5. detect_reverse_items()# Statistical analysis
6. build_r_syntax()    # R code generation
7. Utility functions   # Logging, file writing
```

## Data Classes and Models

### VariableInfo
Represents individual survey variables with metadata:
```python
@dataclass
class VariableInfo:
    name: str              # Variable name
    label: str | None      # SPSS variable label
    item_text: str | None  # Question text (future use)
    missing_pct: float     # Percentage of missing data
    type: Literal["numeric", "ordinal", "string"]  # Variable type
```

### Scale
Represents psychological constructs/scales:
```python
@dataclass
class Scale:
    name: str              # Construct name (e.g., "Anxiety")
    items: List[str]       # Variable names in this scale
    confidence: float      # AI confidence in grouping
    note: str | None       # Additional notes
```

### PromptConfig
Configuration for AI language model:
```python
@dataclass
class PromptConfig:
    system_prompt: str     # Instructions for AI
    temperature: float     # Creativity vs consistency
    top_p: float          # Token selection parameter
```

### Options
Analysis configuration settings:
```python
@dataclass
class Options:
    include_efa: bool                    # Include factor analysis
    missing_strategy: str                # How to handle missing data
    correlation_type: str                # Type of correlation matrix
    reverse_threshold: float             # Threshold for reverse detection
```

## Key Functions Deep Dive

### 1. load_sav(path_or_file)
**Purpose**: Loads SPSS .sav files and extracts metadata
**Features**:
- Handles both file paths and uploaded file objects
- Creates temporary files for BytesIO objects
- Extracts comprehensive metadata including variable labels, types, and missing value ranges
- Returns both DataFrame and structured metadata dictionary

### 2. sanitize_metadata(df, meta)
**Purpose**: Cleans data and removes problematic variables
**Logic**:
- Drops high-cardinality string variables (>50 unique values)
- Removes common ID/timestamp columns
- Applies SPSS missing value ranges as pandas NA
- Prunes all metadata dictionaries to match cleaned dataset

### 3. gemini_detect_scales(var_view, prompt_cfg)
**Current State**: Dummy implementation using variable name prefixes
**Intended Function**: Send variable information to Gemini AI to intelligently group items into psychological constructs
**Output**: List of Scale objects with AI-determined groupings

### 4. detect_reverse_items(scales_confirmed, df)
**Purpose**: Automatically detect items that should be reverse-scored
**Method**:
- Calculates scale totals for each construct
- Computes item-total correlations
- Flags items with negative correlations as reverse items
**Threshold**: Configurable via Options.reverse_threshold

### 5. build_r_syntax(sav_path, scales, rev_map, opts)
**Purpose**: Generates comprehensive R analysis script
**Includes**:
- Package installation and data loading
- Reverse scoring transformations
- Cronbach's α and McDonald's ω reliability analysis
- Optional Exploratory Factor Analysis
- Descriptive statistics and correlation matrices
- Data export functionality

## User Interface Flow

### Sidebar Configuration
- **Analysis Options**: EFA inclusion, missing data strategy, correlation type
- **Reverse Detection**: Threshold slider for sensitivity
- **Real-time Updates**: Changes affect final R script generation

### Main Interface Sequence
1. **File Upload**: Drag-and-drop .sav file interface
2. **Variable Overview**: Tabular display of all variables with metadata
3. **AI Construct Proposals**: Expandable cards showing detected scales
4. **Interactive Editing**: 
   - Modify construct names
   - Add/remove items from scales
   - Toggle construct inclusion
5. **Script Generation**: One-click R script creation and download

### Session State Management
- Caches AI results to prevent re-computation
- Maintains user modifications across interactions
- Preserves state during UI updates

## Current Limitations

### 1. AI Integration
- **Dummy Implementation**: Gemini detection is currently a placeholder
- **No Real Intelligence**: Uses simple name-based clustering
- **Missing API Integration**: No actual language model calls

### 2. Statistical Analysis
- **Basic Reverse Detection**: Only uses correlation-based method
- **Limited EFA Options**: Fixed parameters, no model fit assessment
- **No Advanced Diagnostics**: Missing reliability confidence intervals, factor loadings interpretation

### 3. Data Handling
- **Missing Data**: Limited strategies, no multiple imputation
- **Variable Types**: Simplistic type detection
- **Large Files**: No streaming or memory optimization

### 4. User Experience
- **No Undo/Redo**: Changes to constructs cannot be easily reversed
- **Limited Validation**: No checking for statistical appropriateness
- **No Export Options**: Only R script output, no other formats

### 5. Error Handling
- **Limited Try/Catch**: Basic error handling in core functions
- **No User Feedback**: Errors may not be clearly communicated
- **No Logging**: Missing comprehensive error and usage logging

## Improvement Recommendations

### High Priority Improvements

#### 1. Implement Real AI Integration
```python
# Replace dummy implementation with actual Gemini API calls
async def gemini_detect_scales(var_view: List[VariableInfo], prompt_cfg: PromptConfig) -> List[Scale]:
    """
    Send variable metadata to Gemini for intelligent construct detection
    Include variable names, labels, and descriptive statistics
    Use structured prompts for consistent JSON responses
    """
```

#### 2. Enhanced Error Handling and Logging
```python
import logging
from typing import Optional, Union

def safe_load_sav(path_or_file) -> Union[Tuple[pd.DataFrame, Dict], Tuple[None, None]]:
    """
    Robust file loading with comprehensive error handling
    """
    try:
        return load_sav(path_or_file)
    except Exception as e:
        logging.error(f"Failed to load SPSS file: {e}")
        st.error(f"Error loading file: {e}")
        return None, None
```

#### 3. Advanced Statistical Features
```python
def enhanced_reverse_detection(scales: List[Scale], df: pd.DataFrame, 
                             options: Options) -> Dict[str, Dict]:
    """
    Multi-method reverse item detection:
    - Item-total correlation
    - Factor loading direction
    - Response pattern analysis
    - Statistical significance testing
    """
    return {
        item: {
            'is_reverse': bool,
            'confidence': float,
            'method': str,
            'statistics': dict
        }
    }
```

#### 4. Improved User Interface
```python
# Add these features to main.py:
# - Undo/Redo stack for construct modifications
# - Real-time validation of statistical assumptions
# - Progress indicators for long operations
# - Export options (SPSS, CSV, JSON metadata)
# - Batch processing for multiple files
```

### Medium Priority Improvements

#### 5. Configuration Management
```python
@dataclass
class AppConfig:
    """Centralized configuration management"""
    ai_provider: str = "gemini"
    api_key: Optional[str] = None
    max_file_size: int = 100_000_000  # 100MB
    cache_timeout: int = 3600  # 1 hour
    default_options: Options = field(default_factory=Options)
```

#### 6. Data Validation and Quality Checks
```python
def validate_dataset(df: pd.DataFrame, meta: Dict) -> List[str]:
    """
    Comprehensive data quality assessment:
    - Missing data patterns
    - Outlier detection
    - Variable distribution assessment
    - Sample size adequacy
    - Multicollinearity checks
    """
```

#### 7. Advanced R Script Generation
```python
def build_enhanced_r_syntax(sav_path: str, scales: List[Scale], 
                           analysis_config: Dict) -> str:
    """
    Generate more sophisticated R scripts:
    - Confirmatory Factor Analysis (CFA)
    - Structural Equation Modeling preparation
    - Multiple imputation handling
    - Bootstrap confidence intervals
    - Publication-ready tables and plots
    """
```

### Low Priority Improvements

#### 8. Performance Optimization
- Implement lazy loading for large datasets
- Add caching for expensive computations
- Optimize pandas operations
- Add progress bars for long operations

#### 9. Testing Infrastructure
```python
# Add comprehensive test suite:
# tests/test_pipeline.py
# tests/test_ai_integration.py  
# tests/test_statistical_methods.py
# tests/test_ui_components.py
```

#### 10. Documentation and Help System
- In-app help tooltips
- Statistical method explanations
- Best practices guidance
- Video tutorials integration

### Security and Deployment Improvements

#### 11. Security Enhancements
```python
# Add security measures:
# - Input validation and sanitization
# - File size and type restrictions
# - Rate limiting for AI API calls
# - Secure credential management
```

#### 12. Production Deployment
```python
# Add deployment configurations:
# - Docker containerization
# - Environment variable management  
# - Health check endpoints
# - Monitoring and alerting
```

## Technical Requirements

### Current Dependencies
```
streamlit >= 1.28.0
pandas >= 1.5.0
pyreadstat >= 1.2.0
typing_extensions >= 4.0.0
```

### Recommended Additional Dependencies
```
google-generativeai >= 0.3.0  # For Gemini integration
scipy >= 1.9.0                # Advanced statistical functions
plotly >= 5.0.0               # Interactive visualizations
pydantic >= 2.0.0             # Data validation
redis >= 4.0.0                # Caching (optional)
pytest >= 7.0.0               # Testing framework
```

### System Requirements
- **Python**: 3.8+
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: Minimal (temporary file creation)
- **Network**: Required for AI API calls

## Conclusion

The Survey-to-R Agent represents a sophisticated solution to a common research workflow challenge. While the current implementation provides a solid foundation, the suggested improvements would transform it into a production-ready, enterprise-grade tool for psychological and social science research.

The most critical next step is implementing the real AI integration to unlock the tool's primary value proposition. Combined with enhanced error handling and statistical features, this would create a powerful research accelerator that could significantly impact how survey data analysis is conducted in academic and applied research settings.