# Survey-to-R Agent
**Automated SPSS to R Syntax Converter with LLM-Powered Construct Detection**

The **Survey-to-R Agent** is a powerful tool designed to streamline the process of analyzing survey data. It takes an SPSS (`.sav`) file as input, uses Large Language Models (LLMs) to automatically detect psychological constructs (scales), and generates a complete, ready-to-run R syntax file for statistical analysis.

## Key Features

*   **Automated Construct Detection**: Uses Google Gemini or OpenRouter (e.g., GPT-4, Claude) to analyze variable labels and group them into psychological constructs.
*   **SPSS Integration**: Directly reads `.sav` files, preserving metadata like variable labels and value labels.
*   **Interactive Review**: Provides a user-friendly Streamlit interface to review, modify, and confirm the proposed constructs.
*   **Smart R Syntax Generation**:
    *   **Reliability Analysis**: Calculates Cronbach's alpha for each scale.
    *   **Reverse Item Detection**: Automatically identifies and reverses items that negatively correlate with the scale.
    *   **Exploratory Factor Analysis (EFA)**: Generates code for EFA to validate factor structure.
    *   **Composite Scores**: Creates mean scores for confirmed scales.
    *   **Correlation Matrix**: Generates correlation tables for the scales.
*   **Flexible Configuration**: Choose between different missing value strategies (listwise, pairwise, mean imputation) and correlation types (Pearson, Spearman, Polychoric).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd agentR
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run main.py
    ```

2.  **Configure LLM**:
    *   In the sidebar, select your **Provider** (Gemini or OpenRouter).
    *   Enter your **API Key**.
    *   (Optional) Specify a **Model Name** (e.g., `google/gemini-2.0-flash-001` for OpenRouter).

3.  **Upload Data**:
    *   Upload your SPSS (`.sav`) file using the file uploader.

4.  **Review Constructs**:
    *   The agent will analyze the variables and propose constructs.
    *   Review the proposed scales in the "Proposed Constructs" section.
    *   Rename scales, add/remove items, or uncheck constructs you don't want to keep.

5.  **Generate R Syntax**:
    *   Configure analysis options in the sidebar (EFA, missing strategy, etc.).
    *   Click **"Generate R Syntax"**.
    *   Download the generated `.R` file.

## Configuration

### Environment Variables
You can set default configuration values using environment variables or a `.env` file:

*   `SURVEY_TO_R_LOG_FILE`: Path to the log file (default: `session_log.jsonl`).
*   `SURVEY_TO_R_OUTPUT_DIR`: Directory for output files (default: `outputs`).
*   `SURVEY_TO_R_DEFAULT_PROMPT`: System prompt for the LLM.
*   `SURVEY_TO_R_TEMPERATURE`: Default temperature for LLM generation (default: `0.2`).
*   `GEMINI_API_KEY`: Default API key for Google Gemini.

### Analysis Options
*   **Include EFA**: Generates code for Exploratory Factor Analysis.
*   **Missing Value Strategy**:
    *   `listwise`: Exclude cases with any missing values in the scale.
    *   `pairwise`: Use available data for each pair of variables.
    *   `mean_scale`: Compute scale score if a certain percentage of items are present (default logic in generated R code).
*   **Correlation Type**: Pearson, Spearman, or Polychoric.
*   **Min Δα to reverse item**: Threshold for Cronbach's alpha improvement to justify reversing an item.

## Project Structure

*   `main.py`: Entry point for the Streamlit application.
*   `survey_to_r/`: Core package directory.
    *   `gemini.py`: LLM integration (Gemini & OpenRouter).
    *   `io.py`: File I/O (SPSS loading, R writing).
    *   `analysis.py`: Statistical analysis helper functions.
    *   `r_syntax.py`: R code generation logic.
    *   `models.py`: Data models (Pydantic/Dataclasses).
*   `tests/`: Unit tests.

## Requirements

*   Python 3.8+
*   streamlit
*   pandas
*   pyreadstat (for SPSS files)
*   google-generativeai
*   openai
