# Features and Functionality of Survey-to-R Agent

This document provides a detailed overview of all features and functionality available in the Survey-to-R Agent application.

## Core Features

### 1. SPSS Data Loading and Processing

**File Support**: The application supports SPSS (.sav) files through the pyreadstat library, which allows for:
- Reading both data and metadata from SPSS files
- Preserving variable labels, value labels, and missing value definitions
- Handling different SPSS variable types (numeric, string, date, etc.)

**Metadata Sanitization**: The application automatically cleans and sanitizes the loaded data:
- Identifies and removes non-survey variables (ID fields, timestamps, dates)
- Handles missing value definitions from SPSS
- Filters variables with too many unique values (>50) that are likely free-text responses
- Creates safe variable names for R compatibility

### 2. AI-Powered Construct Detection

**OpenRouter Integration**: Uses OpenRouter API to automatically identify psychological constructs from survey items:
- Analyzes variable labels and item text to group related items
- Assigns confidence scores to each proposed construct
- Provides notes explaining the reasoning for groupings
- Uses customizable prompts and parameters (temperature, top_p)

**Prompt Configuration**: Allows fine-tuning of the AI behavior through:
- Customizable system prompts
- Temperature settings (0.0-1.0) to control creativity/randomness
- Top-p settings (0.0-1.0) for nucleus sampling

### 3. Interactive Scale Management

**Construct Review Interface**: Web-based interface for reviewing and adjusting AI-proposed constructs:
- Expandable panels for each proposed construct
- Ability to rename constructs
- Checkbox to include/exclude constructs
- Multiselect controls for adjusting which items belong to each construct

**Scale Data Model**: Structured representation of psychological scales:
- Name of the construct
- List of constituent items
- Confidence score from AI detection
- Optional notes explaining the grouping

### 4. Statistical Analysis Features

**Reliability Analysis**: Comprehensive reliability testing for each scale:
- Cronbach's Alpha (α) using the `psych` R package
- McDonald's Omega (ω) using the `MBESS` R package
- Check for scale direction (key reversal)

**Reverse Item Detection**: Algorithm to identify items requiring reverse scoring:
- Statistical analysis of item correlations
- Automatic identification of items to reverse-score
- Customizable threshold for detection sensitivity

**Correlation Analysis**: Multiple correlation methods supported:
- Pearson correlation (default)
- Spearman rank correlation
- Polychoric correlation for ordinal data

**Exploratory Factor Analysis (EFA)**: Optional factor analysis capabilities:
- Parallel analysis to determine optimal number of factors
- Factor extraction with various methods
- Promax rotation for correlated factors
- Can be toggled on/off in the UI

### 5. R Syntax Generation

**Complete Analysis Pipeline**: Generates a complete R script including:
- Data loading and preprocessing
- Reverse-scoring operations for identified items
- Reliability analysis for all scales
- Descriptive statistics
- Correlation matrices
- EFA if enabled
- Data export functions

**R Package Integration**: The generated code utilizes several R packages:
- `haven` for SPSS data import
- `tidyverse` for data manipulation
- `psych` for psychological analysis
- `MBESS` for reliability estimation
- `GPArotation` for factor rotation
- `lavaan` for structural equation modeling

## Advanced Features

### 6. Configuration Options

**Analysis Parameters**: Customizable analysis settings:
- Missing data handling strategies (listwise, pairwise, mean imputation)
- Correlation method selection
- EFA inclusion/exclusion
- Reverse item detection threshold

**Security Features**: Built-in security measures:
- Maximum file size limits (default 50MB)
- Path traversal prevention
- Safe string escaping for variable names
- Temporary file cleanup

### 7. Data Validation and Quality Checks

**Data Quality Assessment**: The application performs several quality checks:
- Missing value percentage calculations for each variable
- Automatic detection of non-survey variables
- Validation of data types
- Detection of problematic values or formats

**Error Handling**: Comprehensive error handling for:
- Invalid SPSS files
- File size limits
- Corrupted data
- API connection issues

### 8. Logging and Session Management

**Session Tracking**: The application logs important events:
- File uploads and processing
- Analysis parameters used
- AI detection results
- R script generation events

**Audit Trail**: Maintains an audit trail in JSONL format for:
- Reproducibility of analyses
- Debugging issues
- Tracking usage patterns

## Technical Architecture

### 9. Module Structure

The application is organized into several modules:

**models.py**: Data models and type definitions
- VariableInfo: Represents survey variable metadata
- Scale: Represents psychological constructs
- PromptConfig: Configuration for AI prompting
- Options: Analysis options

**io.py**: Input/output operations
- Loading SPSS files using pyreadstat
- Sanitizing metadata
- Writing output files

**analysis.py**: Statistical analysis functions
- Variable summarization
- Reverse item detection
- Scale validation

**llm.py**: Language model integration
- OpenRouter API integration
- Construct detection algorithms
- Prompt management

**r_syntax.py**: R code generation
- Template-based R script building
- Safe string escaping
- Code formatting

**config.py**: Configuration management
- Environment variable handling
- Configuration validation
- Settings management

**utils.py**: Utility functions
- Session logging
- Pipeline orchestration
- Helper functions

### 10. Web Interface Features

**Streamlit UI**: Modern web interface built with Streamlit:
- Responsive design for different screen sizes
- Interactive widgets for parameter adjustment
- Real-time feedback during processing
- Download buttons for generated files

**Progress Indicators**: Visual feedback during long operations:
- Loading spinners during file processing
- Progress indicators for AI operations
- Success/error messages for user actions

## Algorithm Details

### 11. Construct Detection Algorithm

The AI-powered construct detection uses natural language processing to identify related survey items:

**Input Processing**: 
- Extracts variable names, labels, and item text
- Prepares a structured view of all survey items
- Formats data for AI processing

**AI Processing**:
- Uses system prompt to guide the AI toward psychological construct identification
- Applies temperature and top_p settings for controlled randomness
- Generates confidence scores for each grouping

**Post-processing**:
- Validates generated groupings for logical consistency
- Calculates confidence scores for each construct
- Provides explanatory notes

### 12. Reverse Item Detection Algorithm

The reverse item detection uses correlation analysis:

**Correlation Matrix**: 
- Computes correlations among items and scale totals
- Identifies items with negative or low correlations
- Calculates change in Cronbach's α if item is reversed

**Threshold Application**:
- Compares α difference to configurable threshold
- Flags items exceeding the threshold
- Creates mapping of items to reverse

## Output Specifications

### 13. Generated R Script Features

The R script generated by the application includes:

**Header Information**:
- Generation timestamp
- Tool identification
- Brief description of contents

**Package Management**:
- Automatic package installation/check
- Required package list (haven, tidyverse, psych, MBESS, GPArotation, lavaan)

**Data Preprocessing**:
- SPSS file loading with haven package
- Variable selection and filtering
- Missing value handling according to specified strategy

**Reverse Scoring**:
- Mathematical reversal (new_value = max + min - original_value)
- Application to flagged items only
- Safeguarded against data type issues

**Reliability Analysis**:
- Scale creation for each construct
- Cronbach's α calculation for each scale
- McDonald's Ω calculation for each scale

**Descriptive Statistics**:
- Summary statistics for all variables
- Handling of missing data
- Output to console and saved objects

**Correlation Analysis**:
- Correlation matrix based on selected method
- Handling of missing data according to strategy
- Output to console and saved objects

**EFA (if enabled)**:
- Parallel analysis for factor number determination
- Factor extraction with specified rotation
- Extraction results saved for further analysis

**Export Functions**:
- Save processed dataset as SPSS file
- Save analysis objects as RData file
- Proper file path handling

### 14. File Outputs

**R Script**: The primary output containing complete analysis syntax
**Enhanced Dataset**: SPSS file with reverse-scored items
**Analysis Objects**: RData file containing descriptives and correlations
**Session Logs**: JSONL file containing processing information

## Performance Considerations

### 15. Scalability Features

**Memory Management**:
- Efficient data processing to handle large datasets
- Temporary file cleanup to prevent storage issues
- Streaming processing where applicable

**API Usage Optimization**:
- Minimized API calls for cost efficiency
- Efficient batch processing where possible
- Caching of results to prevent redundant processing

**Processing Time Optimization**:
- Parallel processing where applicable
- Efficient algorithms for statistical calculations
- Progress indicators for long operations

## Security Features

### 16. Data Protection

**File Security**:
- Automatic deletion of temporary files
- File size limits to prevent DoS attacks
- Path validation to prevent traversal attacks

**API Security**:
- Secure handling of API keys
- Input validation for all data
- Safe escaping of variable names and values

**Privacy Considerations**:
- Data is processed on-demand and not stored
- No permanent data retention except in outputs
- User control over data processing

This comprehensive feature set makes the Survey-to-R Agent a powerful tool for researchers transitioning from SPSS to R environments, providing both automation and user control over the analysis process.