# Installation Guide for Survey-to-R Agent

This guide provides detailed instructions for setting up the Survey-to-R Agent application on different operating systems.

## Prerequisites

Before installing the Survey-to-R Agent, ensure you have the following:

1. **Python 3.8 or higher** - The application requires Python 3.8 or later.
2. **Git** (optional but recommended) - For cloning the repository.
3. **Access to an AI API** - OpenRouter API for scale detection functionality.

## Step-by-Step Installation

### Step 1: Obtain the Source Code

Choose one of the following methods:

**Option A: Clone from Git**

```bash
git clone <repository-url>
cd agentR
```

**Option B: Download as ZIP**

1. Download the repository as a ZIP file
2. Extract the contents to your desired location
3. Navigate to the extracted folder

### Step 2: Set Up Virtual Environment (Recommended)

Creating a virtual environment isolates the application's dependencies from your system's Python environment:

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

With your virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install all required dependencies including:
- Streamlit (web interface)
- Pyreadstat (SPSS file reading)
- Pandas (data manipulation)
- NumPy (numerical operations)
- SciPy (scientific computing)
- Scikit-learn (machine learning)
- Pingouin (statistical analysis)
- Factor Analyzer (factor analysis)
- OpenAI (AI integration)
- Pytest (testing)

### Step 4: Configure API Access

The application requires access to an AI service to detect psychological constructs. You will need to set up environment variables:

**For OpenRouter API:**
```bash
# Windows Command Prompt:
set OPENROUTER_API_KEY=your-api-key-here

# Windows PowerShell:
$env:OPENROUTER_API_KEY="your-api-key-here"

# macOS/Linux:
export OPENROUTER_API_KEY='your-api-key-here'
```

**Alternative: Add to Environment File**
You can also create a `.env` file in the project root with your API key:
```
OPENROUTER_API_KEY=your-api-key-here
```

## Troubleshooting Common Installation Issues

### Issue 1: Python Version Compatibility
**Problem:** Error during installation about Python version.
**Solution:** Ensure you're using Python 3.8 or higher:
```bash
python --version
```

### Issue 2: Package Installation Errors
**Problem:** Errors during `pip install -r requirements.txt`
**Solution:** Try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

### Issue 3: Dependency Conflicts
**Problem:** Conflicts between package versions.
**Solution:** Create a fresh virtual environment and try again.

### Issue 4: Missing Visual C++ Build Tools (Windows)
**Problem:** Errors installing packages that require compilation.
**Solution:** Install Microsoft C++ Build Tools from the Microsoft website.

## Verification Steps

After completing the installation:

1. Verify the environment is activated and dependencies are installed:
```bash
pip list | grep -E "(streamlit|pyreadstat|pandas)"
```

2. Test that you can run the application:
```bash
streamlit run main.py
```

3. The application should start and open in your default browser at http://localhost:8501

## Running the Application

Once installed, you can run the application using:

```bash
# Make sure your virtual environment is activated
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Run the application
streamlit run main.py
```

To stop the application, press `Ctrl+C` in the terminal where it's running.

## Updating the Application

To update to the latest version:

1. Pull the latest changes (if using Git):
```bash
git pull origin main
```

2. Update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

## Uninstallation

To completely remove the application:

1. Delete the project directory
2. Remove the virtual environment directory (`.venv`)
3. Optionally, remove any API key environment variables you set

## Additional Notes

- The application creates output files in the `outputs` directory by default
- Session logs are stored in `session_log.jsonl` in the project root
- Large SPSS files (over 50MB) will be rejected by default for security reasons
- API usage may incur costs depending on your service provider