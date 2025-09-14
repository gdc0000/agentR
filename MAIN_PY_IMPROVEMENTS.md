# main.py Improvement Recommendations

## Overview
This file contains specific recommendations for improving `main.py`, the Streamlit frontend of the Survey-to-R Agent.

## Critical Issues to Address

### 1. Error Handling and User Feedback
**Current Issue**: Minimal error handling, poor user experience when things go wrong.

**Improvements**:
```python
# Add comprehensive error handling wrapper
def safe_execute(func, *args, error_msg="An error occurred", **kwargs):
    """Wrapper for safe function execution with user-friendly error messages"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        st.info("Please check your file format and try again.")
        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None

# Replace direct function calls with safe versions
with st.spinner("Reading file…"):
    result = safe_execute(load_sav, uploaded_file, "Failed to load SPSS file")
    if result is None:
        return
    df, meta = result
```

### 2. Session State Management
**Current Issue**: Session state is used but not consistently managed.

**Improvements**:
```python
# Add session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if "scales_prop" not in st.session_state:
        st.session_state.scales_prop = None
    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

# Add file change detection
import hashlib
def get_file_hash(uploaded_file):
    """Generate hash of uploaded file to detect changes"""
    if uploaded_file is None:
        return None
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

# Clear cache when file changes
current_hash = get_file_hash(uploaded_file)
if current_hash != st.session_state.uploaded_file_hash:
    st.session_state.scales_prop = None
    st.session_state.processing_complete = False
    st.session_state.uploaded_file_hash = current_hash
```

### 3. UI/UX Enhancements
**Current Issue**: Basic UI with limited user guidance.

**Improvements**:
```python
# Add help and guidance
st.markdown("""
### 📋 How to Use This Tool
1. **Upload** your SPSS (.sav) file using the uploader below
2. **Configure** analysis options in the sidebar
3. **Review** AI-detected constructs and make adjustments
4. **Generate** and download your R analysis script
""")

# Add file validation
def validate_uploaded_file(uploaded_file):
    """Validate uploaded file before processing"""
    if uploaded_file.size > 100_000_000:  # 100MB
        st.error("File too large. Please use files under 100MB.")
        return False
    
    if not uploaded_file.name.lower().endswith('.sav'):
        st.error("Please upload a valid SPSS (.sav) file.")
        return False
    
    return True

# Add progress tracking
def show_progress_steps(current_step):
    """Show user progress through the workflow"""
    steps = ["Upload File", "Process Data", "Review Constructs", "Generate Script"]
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.success(f"✅ {step}")
            elif i == current_step:
                st.info(f"🔄 {step}")
            else:
                st.gray(f"⏳ {step}")
```

### 4. Configuration Management
**Current Issue**: Hardcoded settings mixed throughout the code.

**Improvements**:
```python
# Add configuration class
@dataclass
class UIConfig:
    """UI configuration settings"""
    max_file_size: int = 100_000_000
    default_reverse_threshold: float = 0.05
    cache_ttl: int = 3600
    max_variables_display: int = 1000

# Move sidebar configuration to a function
def render_analysis_options() -> Options:
    """Render analysis options sidebar"""
    st.sidebar.header("⚙️ Analysis Options")
    
    with st.sidebar.expander("Basic Settings", expanded=True):
        include_efa = st.checkbox("Include Exploratory Factor Analysis", value=True)
        missing_strategy = st.selectbox(
            "Missing Data Strategy",
            ["listwise", "pairwise", "mean_scale"],
            help="How to handle missing data in analyses"
        )
    
    with st.sidebar.expander("Advanced Settings"):
        correlation_type = st.selectbox(
            "Correlation Type",
            ["pearson", "spearman", "polychoric"],
            help="Type of correlation matrix to compute"
        )
        reverse_threshold = st.slider(
            "Reverse Item Detection Threshold",
            0.0, 0.2, 0.05, 0.01,
            help="Minimum alpha improvement to flag reverse items"
        )
    
    return Options(
        include_efa=include_efa,
        missing_strategy=missing_strategy,
        correlation_type=correlation_type,
        reverse_threshold=reverse_threshold
    )
```

### 5. Data Display Improvements
**Current Issue**: Basic dataframe display without filtering or sorting.

**Improvements**:
```python
# Enhanced variable view with filtering
def render_variable_view(var_view_df):
    """Render enhanced variable view with filtering options"""
    st.subheader("📊 Variable Overview")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        name_filter = st.text_input("Filter by name", placeholder="Enter variable name...")
    with col2:
        type_filter = st.multiselect("Filter by type", var_view_df["type"].unique())
    with col3:
        sort_by = st.selectbox("Sort by", ["name", "type", "missing_pct"])
    
    # Apply filters
    filtered_df = var_view_df.copy()
    if name_filter:
        filtered_df = filtered_df[filtered_df["name"].str.contains(name_filter, case=False, na=False)]
    if type_filter:
        filtered_df = filtered_df[filtered_df["type"].isin(type_filter)]
    
    # Sort and display
    filtered_df = filtered_df.sort_values(sort_by)
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "missing_pct": st.column_config.ProgressColumn(
                "Missing %",
                help="Percentage of missing values",
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    st.caption(f"Showing {len(filtered_df)} of {len(var_view_df)} variables")
```

### 6. Construct Review Interface
**Current Issue**: Basic expander interface with limited functionality.

**Improvements**:
```python
def render_construct_review(scales_prop, var_view_df):
    """Enhanced construct review interface"""
    st.subheader("🎯 Review Detected Constructs")
    
    if not scales_prop:
        st.warning("No constructs detected. Please check your data.")
        return []
    
    # Add bulk operations
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Accept All"):
            for i in range(len(scales_prop)):
                st.session_state[f"keep_{i}"] = True
    with col2:
        if st.button("❌ Reject All"):
            for i in range(len(scales_prop)):
                st.session_state[f"keep_{i}"] = False
    with col3:
        min_items = st.number_input("Minimum items per construct", 2, 10, 3)
    
    confirmed_scales = []
    
    for i, scale in enumerate(scales_prop):
        # Enhanced construct display
        confidence_emoji = "🟢" if scale.confidence > 0.8 else "🟡" if scale.confidence > 0.5 else "🔴"
        
        with st.expander(
            f"{confidence_emoji} **{scale.name}** ({len(scale.items)} items, confidence: {scale.confidence:.1%})",
            expanded=scale.confidence > 0.7
        ):
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                new_name = st.text_input(
                    "Construct Name",
                    scale.name,
                    key=f"name_{i}",
                    help="Enter a descriptive name for this construct"
                )
                
                selected_items = st.multiselect(
                    "Items in This Construct",
                    options=var_view_df["name"].tolist(),
                    default=scale.items,
                    key=f"items_{i}",
                    help="Select which variables belong to this construct"
                )
                
                if scale.note:
                    st.info(f"AI Note: {scale.note}")
            
            with col_right:
                keep = st.checkbox(
                    "Include in Analysis",
                    value=len(scale.items) >= min_items,
                    key=f"keep_{i}"
                )
                
                if len(selected_items) < min_items:
                    st.warning(f"Need at least {min_items} items")
                
                # Show basic stats if available
                if len(selected_items) >= 2:
                    st.metric("Items", len(selected_items))
                    st.metric("AI Confidence", f"{scale.confidence:.1%}")
            
            if keep and len(selected_items) >= min_items:
                confirmed_scales.append(
                    Scale(
                        name=new_name,
                        items=selected_items,
                        confidence=scale.confidence,
                        note=scale.note,
                    )
                )
    
    return confirmed_scales
```

### 7. Script Generation Enhancement
**Current Issue**: Basic script generation with no preview or validation.

**Improvements**:
```python
def render_script_generation(confirmed_scales, clean_df, uploaded_file, opts):
    """Enhanced script generation with preview and validation"""
    st.subheader("🔧 Generate R Analysis Script")
    
    if len(confirmed_scales) == 0:
        st.warning("No constructs confirmed. Please review and confirm at least one construct above.")
        return
    
    # Show summary before generation
    with st.expander("📋 Analysis Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confirmed Constructs", len(confirmed_scales))
        with col2:
            total_items = sum(len(s.items) for s in confirmed_scales)
            st.metric("Total Items", total_items)
        with col3:
            st.metric("Sample Size", len(clean_df))
        
        # Show construct summary table
        summary_data = []
        for scale in confirmed_scales:
            summary_data.append({
                "Construct": scale.name,
                "Items": len(scale.items),
                "Variables": ", ".join(scale.items[:3]) + ("..." if len(scale.items) > 3 else "")
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Generation controls
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "🚀 Generate R Script",
            type="primary",
            use_container_width=True
        )
    with col2:
        preview_mode = st.checkbox("Show Preview", value=False)
    
    if generate_button:
        with st.spinner("Detecting reverse items..."):
            rev_map = safe_execute(
                detect_reverse_items, 
                confirmed_scales, 
                clean_df,
                "Failed to detect reverse items"
            )
            if rev_map is None:
                return
        
        with st.spinner("Building R syntax..."):
            r_script = safe_execute(
                build_r_syntax,
                uploaded_file.name,
                confirmed_scales,
                rev_map,
                opts,
                "Failed to generate R script"
            )
            if r_script is None:
                return
        
        # Show reverse items detected
        reverse_items = [k for k, v in rev_map.items() if v]
        if reverse_items:
            st.info(f"🔄 Detected {len(reverse_items)} reverse-scored items: {', '.join(reverse_items)}")
        
        # Preview option
        if preview_mode:
            st.subheader("📄 Script Preview")
            st.code(r_script, language="r", line_numbers=True)
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"survey_analysis_{timestamp}.R"
        
        st.download_button(
            label="⬇️ Download R Script",
            data=r_script,
            file_name=filename,
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
        
        st.success(f"✅ R script generated successfully! ({len(r_script)} characters)")
        
        # Additional export options
        with st.expander("📤 Additional Export Options"):
            col1, col2 = st.columns(2)
            with col1:
                # Export construct definitions as JSON
                construct_json = json.dumps([asdict(s) for s in confirmed_scales], indent=2)
                st.download_button(
                    "Export Construct Definitions (JSON)",
                    construct_json,
                    f"constructs_{timestamp}.json",
                    "application/json"
                )
            with col2:
                # Export analysis settings
                settings_json = json.dumps(asdict(opts), indent=2)
                st.download_button(
                    "Export Analysis Settings (JSON)",
                    settings_json,
                    f"settings_{timestamp}.json",
                    "application/json"
                )
```

### 8. Main Function Restructuring
**Current Issue**: Single large function with mixed concerns.

**Improvements**:
```python
def main() -> None:
    """Streamlit GUI main entry point - restructured for better organization"""
    
    # Initialize app
    initialize_session_state()
    render_header()
    
    # Sidebar configuration
    opts = render_analysis_options()
    
    # File upload and validation
    uploaded_file = st.file_uploader("📁 Upload SPSS (.sav) File", type=["sav"])
    if uploaded_file is None:
        render_welcome_screen()
        return
    
    if not validate_uploaded_file(uploaded_file):
        return
    
    # Step 1: Data Processing
    show_progress_steps(0)
    df, meta, var_view_df = process_uploaded_file(uploaded_file)
    if df is None:
        return
    
    # Step 2: Variable Overview
    show_progress_steps(1)
    render_variable_view(var_view_df)
    
    # Step 3: AI Detection and Review
    show_progress_steps(2)
    scales_prop = get_or_detect_scales(var_view_df)
    if scales_prop is None:
        return
    
    confirmed_scales = render_construct_review(scales_prop, var_view_df)
    
    # Step 4: Script Generation
    show_progress_steps(3)
    render_script_generation(confirmed_scales, df, uploaded_file, opts)

# Split into smaller functions
def render_header():
    """Render application header and instructions"""
    st.title("🤖 Survey-to-R Agent")
    st.markdown("""
    Transform your SPSS survey data into publication-ready R analysis scripts with AI-powered construct detection.
    
    **Features:**
    - 🧠 AI-powered scale detection
    - 🔄 Automatic reverse item identification  
    - 📊 Comprehensive reliability analysis
    - 📈 Optional factor analysis
    - 📝 Ready-to-run R scripts
    """)

def render_welcome_screen():
    """Show welcome screen when no file is uploaded"""
    st.info("👆 Please upload a SPSS (.sav) file to begin analysis.")
    
    with st.expander("ℹ️ What does this tool do?"):
        st.markdown("""
        This tool helps researchers transition from SPSS to R by:
        
        1. **Loading** your SPSS data with all metadata
        2. **Detecting** psychological constructs using AI
        3. **Identifying** reverse-scored items automatically
        4. **Generating** comprehensive R analysis scripts
        5. **Including** reliability analysis, descriptives, and optional EFA
        """)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return cleaned data"""
    with st.spinner("🔍 Processing file..."):
        # Load file
        result = safe_execute(load_sav, uploaded_file, "Failed to load SPSS file")
        if result is None:
            return None, None, None
        df, meta = result
        
        # Clean data
        result = safe_execute(sanitize_metadata, df, meta, "Failed to clean data")
        if result is None:
            return None, None, None
        clean_df, clean_meta = result
        
        # Summarize variables
        var_view = safe_execute(summarize_variables, clean_meta, "Failed to summarize variables")
        if var_view is None:
            return None, None, None
        
        var_view_df = pd.DataFrame(var_view)
        
        st.success(f"✅ Loaded {len(clean_df)} observations with {len(var_view_df)} variables")
        
        return clean_df, clean_meta, var_view_df

def get_or_detect_scales(var_view_df):
    """Get cached scales or detect new ones"""
    var_view = [VariableInfo(**row) for _, row in var_view_df.iterrows()]
    
    if st.session_state.scales_prop is None:
        with st.spinner("🧠 AI is analyzing your variables..."):
            prompt_cfg = PromptConfig(
                system_prompt="Group survey items into psychological constructs.",
                temperature=0.2,
                top_p=0.9,
            )
            scales_prop = safe_execute(
                gemini_detect_scales,
                var_view,
                prompt_cfg,
                "Failed to detect scales"
            )
            if scales_prop is None:
                return None
            st.session_state.scales_prop = scales_prop
    
    return st.session_state.scales_prop
```

## Implementation Priority

1. **Critical (Implement First)**:
   - Error handling and user feedback
   - File validation
   - Session state management

2. **High Priority**:
   - UI/UX enhancements
   - Progress tracking
   - Enhanced construct review

3. **Medium Priority**:
   - Configuration management
   - Script preview
   - Export options

4. **Nice to Have**:
   - Advanced filtering
   - Bulk operations
   - Additional export formats

These improvements will transform the basic Streamlit app into a professional, user-friendly tool suitable for research environments.