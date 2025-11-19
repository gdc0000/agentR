# Usage Guide for Survey-to-R Agent

This guide provides detailed instructions for using the Survey-to-R Agent application to convert SPSS survey data into R analysis syntax.

## Getting Started

### 1. Launching the Application

1. Ensure you have completed the installation process and activated your virtual environment
2. Run the application:
   ```bash
   streamlit run main.py
   ```
3. The application will open in your default web browser at `http://localhost:8501`
4. You should see the main interface with the title "Survey‑to‑R Agent"

### 2. Preparing Your SPSS File

Before uploading, ensure your SPSS file (.sav) meets these requirements:

- Contains survey data with properly labeled variables
- Variable labels include the actual survey item text (recommended)
- File size is under 50MB (default limit)
- Uses numeric or ordinal data types for survey responses
- Does not contain sensitive information that should not be processed by AI

## Step-by-Step Workflow

### Step 1: Configure Analysis Options (Sidebar)

On the left sidebar, you'll find several analysis options:

- **Include EFA**: Toggle whether to include Exploratory Factor Analysis in the R output
- **Missing Strategy**: Choose how to handle missing data:
  - `listwise`: Remove entire cases with any missing values
  - `pairwise`: Use all available data for each analysis
  - `mean_scale`: Replace missing values with scale mean
- **Correlation Type**: Select the correlation method:
  - `pearson`: Standard Pearson correlation
  - `spearman`: Rank-order correlation
  - `polychoric`: For ordinal data (requires additional package in R)
- **Δα minimum for reverse item**: Set the minimum alpha difference threshold for identifying items that need reverse-scoring

Adjust these options according to your analysis needs.

### Step 2: Upload Your SPSS File

1. Click the "Carica file .sav" (Upload .sav file) button
2. Select your SPSS (.sav) file from your local storage
3. Wait for the file to upload and load
4. The application will display a message when loading is complete

### Step 3: Review Variable Information

After loading, the "Variable view" section will show a table with information about each variable in your dataset:

- **name**: The variable name in the SPSS file
- **label**: The variable label (should contain survey item text)
- **item_text**: Extracted item text (if available)
- **missing_pct**: Percentage of missing values
- **type**: Data type (numeric, ordinal, string)

Review this information to ensure all relevant survey items are present and properly identified.

### Step 4: AI-Proposed Constructs Review

The application will automatically use AI to propose psychological constructs based on your survey items. This may take a minute depending on the number of items.

Each proposed construct will appear in an expandable section:

- **Construct name**: The AI-proposed name for the construct
- **Number of items**: How many survey items are included in this construct

For each proposed construct, you can:

1. **Modify the name**: Change the construct name using the text input field
2. **Include/Exclude**: Toggle the checkbox to confirm whether to keep this construct
3. **Adjust items**: Use the multiselect dropdown to add or remove specific items from the construct

Take time to review and adjust the constructs as needed. You may:
- Combine multiple small constructs
- Split large constructs that group unrelated items
- Rename constructs to match your theoretical framework
- Remove constructs that are not relevant to your analysis

### Step 5: Generate R Syntax

Once you're satisfied with the confirmed constructs:

1. Click the "Genera sintassi R" (Generate R syntax) button
2. The application will:
   - Determine which items need reverse-scoring
   - Build the complete R analysis script
3. A download button titled "Scarica sintassi R" (Download R syntax) will appear
4. The generated script includes:
   - Data loading and preprocessing
   - Reverse-scoring operations for flagged items
   - Reliability analysis (Cronbach's α and McDonald's Ω) for each scale
   - Exploratory Factor Analysis (if enabled)
   - Descriptive statistics and correlation matrices
   - Functions to save processed data and results

### Step 6: Download and Use the R Script

1. Click the download button to save the R script to your computer
2. The file will be named with a timestamp (e.g., `analysis_20231117103045.R`)
3. Open the script in your R environment (RStudio, R console, etc.)
4. The script will require R packages: `haven`, `tidyverse`, `psych`, `MBESS`, `GPArotation`, `lavaan`
5. Install any missing packages before running the analysis

## Detailed Feature Descriptions

### Reverse Item Detection

The application automatically identifies survey items that may need to be reverse-scored. This is done by analyzing the correlation patterns between items and scales. Items flagged for reversal will be mathematically reversed (e.g., 1 becomes 5, 2 becomes 4) in the final R script.

The threshold for flagging reverse items can be adjusted using the "Δα minimo per invertire item" slider in the sidebar. A lower threshold (e.g., 0.01) will flag more items, while a higher threshold (e.g., 0.10) will be more conservative.

### Reliability Analysis

For each confirmed scale, the R script will generate:
- Cronbach's Alpha (α) for internal consistency
- McDonald's Omega (ω) for reliability assessment

These measures are calculated using the `psych` and `MBESS` R packages.

### Exploratory Factor Analysis (EFA)

If enabled, the script will perform parallel analysis using `psych::fa.parallel()` to determine the number of factors to extract. It will then perform factor analysis with the `psych::fa()` function using promax rotation.

### Data Handling

The application:
- Automatically removes ID variables, timestamps, and other non-survey variables
- Properly handles SPSS missing value definitions
- Preserves value labels and variable metadata
- Safely escapes variable names to prevent R syntax errors

## Tips for Best Results

1. **Quality of SPSS files**: Ensure your SPSS file has informative variable labels that contain the actual survey items. This helps the AI better understand and group the items.

2. **Scale detection**: Review AI-proposed constructs carefully. The AI may group items based on keywords rather than theoretical constructs, so human validation is important.

3. **File size**: Keep your SPSS files under 50MB for optimal performance. Large files may take longer to process and may hit API limits.

4. **Variable naming**: Use clear, descriptive variable names in your SPSS file to make the final R script easier to interpret.

5. **AI prompts**: The application uses an AI system prompt to guide construct detection. In the code, you can customize the `PromptConfig` for different approaches to construct identification.

## Troubleshooting Common Issues

### Problem: Upload fails or times out
- **Solution**: Check that your file is under 50MB and is a valid .sav file
- **Solution**: Restart the application and try again

### Problem: AI doesn't propose any constructs
- **Solution**: Verify that your SPSS file has variable labels containing survey items
- **Solution**: Check that your API key is properly configured and has sufficient quota

### Problem: Generated R script doesn't run
- **Solution**: Ensure all required R packages are installed
- **Solution**: Check for special characters in variable names that might cause syntax errors

### Problem: Reverse items not identified correctly
- **Solution**: Adjust the "Δα minimo per invertire item" threshold
- **Solution**: Manually review and adjust in the R script if needed

## Output Files

The application generates several types of output:

1. **R Script**: Contains complete analysis syntax (downloaded from the UI)
2. **Enhanced Dataset**: Saved as `enhanced_dataset.sav` with reverse-scored items
3. **Analysis Objects**: Saved as `analysis_objects.RData` containing descriptives and correlations
4. **Session Logs**: Saved in `session_log.jsonl` for debugging and audit purposes

## Best Practices

1. **Validate AI results**: Always review and validate the AI-proposed constructs before generating R syntax
2. **Document your decisions**: Keep notes on construct modifications for research reproducibility
3. **Test with small datasets**: Try the application with a small subset of your data first
4. **Backup your data**: Always keep original copies of your SPSS files
5. **Review generated code**: Examine the R script before running it with full data