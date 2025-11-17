# Survey-to-R Agent

A Streamlit-based application that transforms SPSS survey data into ready-to-use R analysis syntax. The application leverages AI to detect psychological constructs from survey items and generates comprehensive R scripts for statistical analysis including reliability testing, correlations, and exploratory factor analysis.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- **SPSS File Loading**: Load and parse SPSS (.sav) files with full metadata preservation
- **AI-Powered Scale Detection**: Uses OpenRouter API (supporting multiple AI models) to identify and group survey items into psychological constructs
- **Interactive Scale Confirmation**: Review and adjust AI-proposed constructs with a user-friendly interface
- **Reverse Item Detection**: Automatically identifies items that need to be reverse-scored
- **Customizable Analysis Options**:
  - Missing data strategies (listwise, pairwise, mean imputation)
  - Correlation types (Pearson, Spearman, Polychoric)
  - EFA inclusion/exclusion
  - Customizable thresholds for reverse item detection
- **Complete R Script Generation**: Outputs ready-to-run R syntax with reliability analysis, correlations, and EFA
- **Streamlit Web Interface**: Easy-to-use web-based interface for non-programmers

## Installation

### Prerequisites
- Python 3.8 or higher
- Access to OpenRouter API for AI-powered construct detection

### Steps

1. Clone or download the repository:
```bash
git clone <repository-url>
cd agentR
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key environment variables:
```bash
# For OpenRouter API
export OPENROUTER_API_KEY='your-api-key'
```

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run main.py
```

The web interface will open automatically in your browser (typically at http://localhost:8501).

### Step-by-Step Process

1. **Upload SPSS File**: Upload your survey data in SPSS (.sav) format
2. **Adjust Analysis Options** (sidebar):
   - Include EFA: Toggle whether to include Exploratory Factor Analysis
   - Missing Strategy: Choose between 'listwise', 'pairwise', or 'mean_scale' imputation
   - Correlation Type: Select 'pearson', 'spearman', or 'polychoric' correlation
   - Reverse Threshold: Set minimum alpha difference to flag reverse items
3. **Review Variable View**: Examine the loaded variables in the dataset
4. **Review AI-Proposed Constructs**: The app will automatically propose psychological constructs based on the survey items
   - Rename constructs as needed
   - Select/deselect constructs to include
   - Adjust which items belong to each construct
5. **Generate R Script**: Click the "Generate R syntax" button to create the analysis script
6. **Download R Syntax**: Download the generated R script and run it in your R environment

## Configuration

The application can be configured through environment variables:

- `SURVEY_TO_R_LOG_FILE`: Path to the session log file (default: "session_log.jsonl")
- `SURVEY_TO_R_OUTPUT_DIR`: Directory for output files (default: "outputs")
- `SURVEY_TO_R_DEFAULT_PROMPT`: System prompt for AI (default: "Group survey items into psychological constructs.")
- `SURVEY_TO_R_TEMPERATURE`: AI temperature (default: 0.2)
- `SURVEY_TO_R_TOP_P`: AI top_p (default: 0.9)
- `SURVEY_TO_R_MAX_FILE_SIZE`: Maximum file size in MB (default: 50)
- `SURVEY_TO_R_ROOT_OUTPUT_DIR`: Root output directory (default: "outputs")
- `SURVEY_TO_R_MASK_LOGS`: Whether to mask log file names (default: "true")
- `SURVEY_TO_R_ENABLE_LOGGING`: Enable session logging (default: "true")
- `OPENROUTER_API_KEY`: API key for OpenRouter
- `OPENROUTER_MODEL`: Model to use on OpenRouter (default: "openai/gpt-4o-mini") - any OpenRouter-compatible model can be used

## How It Works

### Data Processing Pipeline
1. **File Loading**: Loads SPSS files using pyreadstat, extracting both data and metadata
2. **Metadata Sanitization**: Removes potentially problematic columns (IDs, timestamps) and handles missing values
3. **Variable Summarization**: Creates a summary view of all variables in the dataset
4. **AI Construct Detection**: Uses OpenRouter API (supporting multiple AI models) to group survey items into psychological constructs based on their labels and content
5. **Interactive Review**: Allows users to confirm, rename, and adjust the AI-proposed constructs
6. **Reverse Item Detection**: Analyzes item correlations to identify items that should be reverse-scored
7. **R Syntax Generation**: Creates a comprehensive R script that includes:
   - Data loading and preprocessing
   - Reverse-scoring for identified items
   - Reliability analysis (Cronbach's α and McDonald's Ω)
   - Exploratory Factor Analysis (if enabled)
   - Descriptive statistics and correlations
   - Data export functions

### Key Components
- `main.py`: Streamlit web interface
- `survey_to_r/io.py`: Data input/output operations
- `survey_to_r/analysis.py`: Statistical analysis functions
- `survey_to_r/llm.py`: AI integration for construct detection
- `survey_to_r/r_syntax.py`: R script generation
- `survey_to_r/models.py`: Data models and type definitions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions about the application, please open an issue in the GitHub repository.