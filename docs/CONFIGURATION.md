# Configuration Options for Survey-to-R Agent

This document details all configuration options available in the Survey-to-R Agent application, including environment variables, runtime settings, and API configurations.

## Environment Variables

The application uses environment variables for configuration, which can be set in several ways:

1. Directly in the shell:
   ```bash
   export SURVEY_TO_R_TEMPERATURE=0.5
   ```

2. Using a .env file in the project root directory

3. Through your system's environment variable settings

### Logging Configuration

**SURVEY_TO_R_LOG_FILE**
- Default: `session_log.jsonl`
- Description: Path to the file where session logs will be written
- Format: JSONL (JSON Lines format)
- Content: Processing events, errors, and audit information

**SURVEY_TO_R_ENABLE_LOGGING**
- Default: `true`
- Description: Enable or disable session logging
- Values: `true` or `false`
- When disabled: No session logs will be written to file

**SURVEY_TO_R_MASK_LOGS**
- Default: `true`
- Description: Whether to mask potentially sensitive information in logs
- Values: `true` or `false`
- When enabled: File names and other identifiers may be anonymized

### Output Directories

**SURVEY_TO_R_OUTPUT_DIR**
- Default: `outputs`
- Description: Directory for generated output files
- Creates directory if it doesn't exist
- Used for temporary output files

**SURVEY_TO_R_ROOT_OUTPUT_DIR**
- Default: `outputs`
- Description: Root directory for all output files
- Used to prevent path traversal attacks
- All output directories must be within this root

### AI Model Configuration

**OPENROUTER_API_KEY**
- Default: None
- Description: API key for OpenRouter service
- Required for AI-powered construct detection
- Must have appropriate permissions and quota

**OPENROUTER_MODEL**
- Default: `openai/gpt-4o-mini`
- Description: Model to use on OpenRouter for construct detection
- Possible values: Any model available on OpenRouter that supports the OpenAI API format
- Examples: `openai/gpt-4o-mini`, `openai/gpt-4`, `anthropic/claude-3-opus`, etc.

**SURVEY_TO_R_DEFAULT_PROMPT**
- Default: `Group survey items into psychological constructs.`
- Description: System prompt used to guide AI construct detection
- Customizable to influence the AI's grouping strategy
- Should be phrased as instructions for identifying psychological constructs from survey items

**SURVEY_TO_R_TEMPERATURE**
- Default: `0.2`
- Description: Controls randomness in AI responses
- Range: 0.0 to 1.0
- Lower values (0.0-0.3): More deterministic, consistent outputs
- Higher values (0.7-1.0): More creative, varied outputs
- Recommended for this use case: Low values (0.1-0.3) for consistency

**SURVEY_TO_R_TOP_P**
- Default: `0.9`
- Description: Controls diversity via nucleus sampling
- Range: 0.0 to 1.0
- Lower values: More focused, conservative outputs
- Higher values: More diverse, exploratory outputs
- Recommended: 0.8-0.9 for balanced results


### File Handling Configuration

**SURVEY_TO_R_MAX_FILE_SIZE**
- Default: `50`
- Description: Maximum allowed file size in megabytes
- Purpose: Security measure to prevent DoS attacks from large files
- Unit: Integer value representing MB
- Files exceeding this size will be rejected

## Configuration Through Code

### PromptConfig Model

The application also allows configuration through the PromptConfig data model:

- `system_prompt`: Override the default system prompt
- `temperature`: Override the default temperature setting
- `top_p`: Override the default top_p setting

Example usage in code:
```python
from survey_to_r.models import PromptConfig

prompt_cfg = PromptConfig(
    system_prompt="Group survey items by theoretical framework.",
    temperature=0.3,
    top_p=0.8,
)
```

### Options Model

Runtime analysis options are configurable via the Options data model:

- `include_efa`: Whether to include Exploratory Factor Analysis (default: True)
- `missing_strategy`: Strategy for handling missing values (default: "listwise")
  - Available values: "listwise", "pairwise", "mean_scale"
- `correlation_type`: Type of correlation to calculate (default: "pearson")
  - Available values: "pearson", "spearman", "polychoric"
- `reverse_threshold`: Minimum α difference to flag reverse items (default: 0.05)

Example usage in code:
```python
from survey_to_r.models import Options

opts = Options(
    include_efa=True,
    missing_strategy="pairwise",
    correlation_type="spearman",
    reverse_threshold=0.02,
)
```

## Configuration Validation

The application validates configuration values during startup and runtime:

1. **Numeric Value Validation**:
   - Temperature must be between 0 and 1
   - Top_p must be between 0 and 1
   - Max file size must be positive

2. **Directory Access Validation**:
   - Output directories must be writable
   - Directory will be created if it doesn't exist but is writable

3. **Environment Variable Type Conversion**:
   - String values are converted to appropriate types (int, float, bool)
   - Invalid conversions result in fallback to default values

## Runtime Configuration Files

While the application primarily uses environment variables, it also supports configuration from JSON files:

### Configuration File Loading

The `load_config_from_file` function allows loading configuration from a JSON file:

```python
from survey_to_r.config import load_config_from_file

success = load_config_from_file('config.json')
```

The JSON file should contain key-value pairs matching the environment variable names:

```json
{
  "SURVEY_TO_R_TEMPERATURE": 0.3,
  "SURVEY_TO_R_TOP_P": 0.85,
  "SURVEY_TO_R_MAX_FILE_SIZE": 100,
  "SURVEY_TO_R_DEFAULT_PROMPT": "Group survey items by domains of interest."
}
```

## Recommended Configuration Values

### For Consistent Results
- `SURVEY_TO_R_TEMPERATURE`: 0.1-0.2 (low randomness for reproducible groups)
- `SURVEY_TO_R_TOP_P`: 0.8-0.9 (good balance of focus and creativity)
- `SURVEY_TO_R_DEFAULT_PROMPT`: Clear, specific instructions for psychological grouping

### For Exploratory Analysis
- `SURVEY_TO_R_TEMPERATURE`: 0.4-0.6 (more diverse groupings)
- `SURVEY_TO_R_TOP_P`: 0.9-1.0 (maximum diversity)
- `SURVEY_TO_R_DEFAULT_PROMPT`: Broader instructions to encourage different perspectives

### For Large Datasets
- `SURVEY_TO_R_MAX_FILE_SIZE`: Adjust based on your system's capabilities
- Consider increasing if working with larger SPSS files

## Security Configuration

### API Key Security
- Store API keys in environment variables, not in code
- Use appropriate permissions for environment files
- Rotate API keys regularly
- Monitor API usage for unusual patterns

### File Security
- Set appropriate file permissions on output directories
- Regularly clean up old log and output files
- Implement network-level security if running on a server

## Configuration Best Practices

1. **Environment-Specific Configurations**:
   - Use different API keys for development and production
   - Adjust file size limits based on your typical data sizes
   - Set appropriate logging levels (more detailed for development, less for production)

2. **Configuration Management**:
   - Document your configuration changes
   - Test configuration changes with sample data before applying to research data
   - Maintain backup configurations for reproducibility

3. **Performance Tuning**:
   - Adjust AI parameters based on your consistency requirements
   - Set file size limits appropriate to your system resources
   - Monitor API usage and adjust accordingly

4. **User Experience**:
   - Choose settings that match your analysis workflow
   - Balance automation with manual review options
   - Consider the trade-off between analysis speed and thoroughness

## Troubleshooting Configuration Issues

### Common Issues

1. **AI API Not Working**:
   - Verify API key is correctly set
   - Check that the API key has sufficient quota
   - Confirm network connectivity to the API provider

2. **Invalid Configuration Values**:
   - Ensure numeric values are within acceptable ranges
   - Verify directory paths exist and are writable
   - Check boolean values are set as "true"/"false" strings

3. **File Size Issues**:
   - If frequently hitting file limits, increase `SURVEY_TO_R_MAX_FILE_SIZE`
   - If memory issues occur, decrease the limit
   - Consider preprocessing large files to reduce size before analysis

This configuration system provides flexibility to adapt the Survey-to-R Agent to different use cases, security requirements, and performance needs while maintaining security and usability.