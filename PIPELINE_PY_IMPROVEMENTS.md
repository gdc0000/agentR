# pipeline.py Improvement Recommendations

## Overview
This file contains specific recommendations for improving `pipeline.py`, the core business logic of the Survey-to-R Agent.

## Critical Issues to Address

### 1. Implement Real AI Integration
**Current Issue**: Dummy implementation that doesn't use actual AI.

**Solution - Real Gemini Integration**:
```python
import google.generativeai as genai
from typing import Optional
import json
import logging

def configure_gemini(api_key: str) -> None:
    """Configure Gemini API with provided key"""
    genai.configure(api_key=api_key)

async def gemini_detect_scales(
    var_view: List[VariableInfo], 
    prompt_cfg: PromptConfig
) -> List[Scale]:
    """
    Use real Gemini AI to detect psychological constructs from variable metadata
    """
    try:
        # Prepare variable information for AI
        variable_info = []
        for var in var_view:
            variable_info.append({
                "name": var.name,
                "label": var.label,
                "type": var.type,
                "missing_pct": var.missing_pct
            })
        
        # Create structured prompt
        prompt = f"""
        {prompt_cfg.system_prompt}
        
        Analyze the following survey variables and group them into psychological constructs/scales.
        Consider variable names, labels, and context to identify related items.
        
        Variables:
        {json.dumps(variable_info, indent=2)}
        
        Return a JSON array of constructs with this exact structure:
        [
            {{
                "name": "construct_name",
                "items": ["var1", "var2", "var3"],
                "confidence": 0.85,
                "note": "Brief explanation of grouping rationale"
            }}
        ]
        
        Guidelines:
        - Group 3-10 related items per construct
        - Use descriptive psychological construct names
        - Confidence should reflect certainty (0.0-1.0)
        - Only include variables that clearly belong together
        - Minimum 2 items per construct
        """
        
        # Configure model
        model = genai.GenerativeModel('gemini-pro')
        generation_config = genai.types.GenerationConfig(
            temperature=prompt_cfg.temperature,
            top_p=prompt_cfg.top_p,
            max_output_tokens=4096,
        )
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse JSON response
        try:
            constructs_data = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # Fallback: extract JSON from response if wrapped in other text
            import re
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                constructs_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse AI response as JSON")
        
        # Convert to Scale objects
        scales = []
        available_vars = {v.name for v in var_view}
        
        for construct in constructs_data:
            # Validate items exist in dataset
            valid_items = [item for item in construct["items"] if item in available_vars]
            
            if len(valid_items) >= 2:  # Minimum 2 items per scale
                scales.append(Scale(
                    name=construct["name"],
                    items=valid_items,
                    confidence=min(max(construct["confidence"], 0.0), 1.0),  # Clamp 0-1
                    note=construct.get("note")
                ))
        
        logging.info(f"AI detected {len(scales)} constructs from {len(var_view)} variables")
        return scales
        
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        # Fallback to rule-based detection
        return _fallback_scale_detection(var_view)

def _fallback_scale_detection(var_view: List[VariableInfo]) -> List[Scale]:
    """Fallback rule-based scale detection when AI fails"""
    # Enhanced version of current dummy implementation
    clusters: Dict[str, List[str]] = {}
    
    for v in var_view:
        # Try multiple clustering strategies
        if "_" in v.name:
            prefix = v.name.split("_")[0]
        elif any(char.isdigit() for char in v.name):
            # Group by non-digit prefix
            prefix = ''.join(c for c in v.name if not c.isdigit())
        else:
            prefix = "misc"
        
        clusters.setdefault(prefix, []).append(v.name)
    
    # Filter out single-item clusters and create scales
    scales = []
    for name, items in clusters.items():
        if len(items) >= 2:  # Only multi-item scales
            scales.append(Scale(
                name=name.title().replace("_", " "),
                items=items,
                confidence=0.3,  # Low confidence for rule-based
                note="Detected using rule-based fallback (consider manual review)"
            ))
    
    return scales
```

### 2. Enhanced Statistical Analysis
**Current Issue**: Basic correlation-based reverse detection only.

**Solution - Multi-Method Reverse Detection**:
```python
from scipy import stats
from typing import Dict, List, Tuple
import numpy as np

def enhanced_reverse_detection(
    scales_confirmed: List[Scale], 
    df: Any,
    methods: List[str] = ["correlation", "alpha_improvement", "factor_loading"]
) -> Dict[str, Dict]:
    """
    Multi-method reverse item detection with comprehensive statistics
    """
    pd = _ensure_pandas()
    reverse_analysis = {}
    
    for scale in scales_confirmed:
        if len(scale.items) < 3:  # Need at least 3 items for reliable analysis
            continue
            
        scale_data = df[scale.items].dropna()
        if len(scale_data) < 10:  # Need sufficient sample
            continue
        
        scale_results = {}
        
        # Method 1: Item-total correlation
        if "correlation" in methods:
            scale_results.update(_correlation_method(scale_data, scale.items))
        
        # Method 2: Alpha improvement method
        if "alpha_improvement" in methods:
            scale_results.update(_alpha_improvement_method(scale_data, scale.items))
        
        # Method 3: Factor loading direction (if EFA possible)
        if "factor_loading" in methods and len(scale.items) >= 4:
            scale_results.update(_factor_loading_method(scale_data, scale.items))
        
        # Combine methods for final decision
        for item in scale.items:
            item_evidence = []
            confidence_scores = []
            
            for method in methods:
                if f"{method}_reverse" in scale_results.get(item, {}):
                    item_evidence.append(scale_results[item][f"{method}_reverse"])
                    confidence_scores.append(scale_results[item][f"{method}_confidence"])
            
            # Majority vote with weighted confidence
            if item_evidence:
                final_reverse = sum(item_evidence) > len(item_evidence) / 2
                final_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
                
                reverse_analysis[item] = {
                    'is_reverse': final_reverse,
                    'confidence': final_confidence,
                    'methods_agree': len(set(item_evidence)) == 1,
                    'evidence_count': sum(item_evidence),
                    'total_methods': len(item_evidence),
                    'details': scale_results.get(item, {})
                }
    
    return reverse_analysis

def _correlation_method(scale_data: pd.DataFrame, items: List[str]) -> Dict:
    """Item-total correlation method"""
    results = {}
    scale_total = scale_data.sum(axis=1)
    
    for item in items:
        # Corrected item-total correlation (remove item from total)
        corrected_total = scale_total - scale_data[item]
        correlation = scale_data[item].corr(corrected_total)
        
        results[item] = {
            'correlation_reverse': correlation < -0.1,  # Threshold for negative correlation
            'correlation_confidence': abs(correlation) if not pd.isna(correlation) else 0.0,
            'correlation_value': correlation
        }
    
    return results

def _alpha_improvement_method(scale_data: pd.DataFrame, items: List[str]) -> Dict:
    """Cronbach's alpha improvement method"""
    results = {}
    
    # Calculate original alpha
    original_alpha = calculate_cronbach_alpha(scale_data)
    
    for item in items:
        # Calculate alpha with item reversed
        test_data = scale_data.copy()
        item_max = test_data[item].max()
        item_min = test_data[item].min()
        test_data[item] = item_max + item_min - test_data[item]
        
        reversed_alpha = calculate_cronbach_alpha(test_data)
        alpha_improvement = reversed_alpha - original_alpha
        
        results[item] = {
            'alpha_reverse': alpha_improvement > 0.05,  # Meaningful improvement
            'alpha_confidence': min(alpha_improvement * 10, 1.0) if alpha_improvement > 0 else 0.0,
            'alpha_improvement': alpha_improvement
        }
    
    return results

def _factor_loading_method(scale_data: pd.DataFrame, items: List[str]) -> Dict:
    """Factor loading direction method"""
    results = {}
    
    try:
        from sklearn.decomposition import FactorAnalysis
        
        # Fit single factor model
        fa = FactorAnalysis(n_components=1, random_state=42)
        fa.fit(scale_data.fillna(scale_data.mean()))
        
        loadings = fa.components_[0]
        
        for i, item in enumerate(items):
            loading = loadings[i]
            results[item] = {
                'factor_reverse': loading < -0.3,  # Negative loading threshold
                'factor_confidence': abs(loading),
                'factor_loading': loading
            }
    
    except ImportError:
        # Fallback if sklearn not available
        for item in items:
            results[item] = {
                'factor_reverse': False,
                'factor_confidence': 0.0,
                'factor_loading': None
            }
    
    return results

def calculate_cronbach_alpha(data: pd.DataFrame) -> float:
    """Calculate Cronbach's alpha reliability coefficient"""
    n_items = data.shape[1]
    if n_items < 2:
        return 0.0
    
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
```

### 3. Enhanced R Script Generation
**Current Issue**: Basic R script with hardcoded parameters and limited analysis options.

**Solution - Comprehensive R Script Builder**:
```python
from enum import Enum
from typing import Optional, Union

class AnalysisType(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class RScriptConfig:
    """Configuration for R script generation"""
    analysis_level: AnalysisType = AnalysisType.INTERMEDIATE
    include_plots: bool = True
    include_assumptions_checks: bool = True
    bootstrap_ci: bool = False
    n_bootstrap: int = 1000
    cfa_model: bool = False
    parallel_analysis: bool = True
    output_format: str = "both"  # "console", "file", "both"

def build_enhanced_r_syntax(
    sav_path: str,
    scales: List[Scale],
    reverse_map: Dict[str, Dict],
    opts: Options,
    config: RScriptConfig = RScriptConfig()
) -> str:
    """
    Generate comprehensive R analysis script with multiple analysis levels
    """
    lines = []
    a = lines.append
    
    # Header with metadata
    a("# " + "="*70)
    a("# Survey-to-R Agent: Enhanced Analysis Script")
    a(f"# Generated: {datetime.now().isoformat()}")
    a(f"# Analysis Level: {config.analysis_level.value}")
    a(f"# Constructs: {len(scales)}")
    a(f"# Total Items: {sum(len(s.items) for s in scales)}")
    a("# " + "="*70)
    a("")
    
    # Package installation and loading
    a("# Package Management")
    packages = _get_required_packages(config, opts)
    a("required_packages <- c(" + ", ".join(f"'{pkg}'" for pkg in packages) + ")")
    a("missing_packages <- required_packages[!(required_packages %in% installed.packages()[,'Package'])]")
    a("if(length(missing_packages)) install.packages(missing_packages)")
    a("invisible(lapply(required_packages, library, character.only = TRUE))")
    a("")
    
    # Data loading with error handling
    a("# Data Loading and Preparation")
    a("tryCatch({")
    a(f"  data <- haven::read_sav('{sav_path}')")
    a("  cat('Data loaded successfully:', nrow(data), 'observations,', ncol(data), 'variables\\n')")
    a("}, error = function(e) {")
    a("  stop('Failed to load data: ', e$message)")
    a("})")
    a("")
    
    # Reverse scoring with detailed logging
    reverse_items = [item for item, info in reverse_map.items() if info.get('is_reverse', False)]
    if reverse_items:
        a("# Reverse Scoring")
        a("reverse_items <- c(" + ", ".join(f"'{item}'" for item in reverse_items) + ")")
        a("reverse_log <- data.frame(")
        a("  item = character(),")
        a("  method = character(),")
        a("  confidence = numeric(),")
        a("  original_mean = numeric(),")
        a("  reversed_mean = numeric()")
        a(")")
        a("")
        
        for item in reverse_items:
            info = reverse_map[item]
            a(f"# Reversing {item} (confidence: {info['confidence']:.2f})")
            a(f"original_mean <- mean(data${item}, na.rm = TRUE)")
            a(f"maxv <- max(data${item}, na.rm = TRUE)")
            a(f"minv <- min(data${item}, na.rm = TRUE)")
            a(f"data${item} <- maxv + minv - data${item}")
            a(f"reversed_mean <- mean(data${item}, na.rm = TRUE)")
            a("reverse_log <- rbind(reverse_log, data.frame(")
            a(f"  item = '{item}',")
            a(f"  method = 'multi_method',")
            a(f"  confidence = {info['confidence']:.3f},")
            a("  original_mean = original_mean,")
            a("  reversed_mean = reversed_mean")
            a("))")
            a("")
    
    # Comprehensive reliability analysis
    a("# Reliability Analysis")
    a("reliability_results <- list()")
    a("")
    
    for scale in scales:
        scale_name = _sanitize_r_name(scale.name)
        items_var = f"items_{scale_name}"
        
        a(f"# {scale.name} ({len(scale.items)} items)")
        a(f"{items_var} <- c(" + ", ".join(f"'{item}'" for item in scale.items) + ")")
        a(f"scale_data <- data[{items_var}]")
        a("")
        
        # Basic reliability
        a(f"alpha_{scale_name} <- psych::alpha(scale_data, check.keys = TRUE)")
        
        if config.analysis_level in [AnalysisType.INTERMEDIATE, AnalysisType.ADVANCED]:
            a(f"omega_{scale_name} <- psych::omega(scale_data, nfactors = 1)")
            
        if config.bootstrap_ci and config.analysis_level == AnalysisType.ADVANCED:
            a(f"alpha_ci_{scale_name} <- psych::alpha.ci(scale_data, n.iter = {config.n_bootstrap})")
        
        a(f"reliability_results${scale_name} <- list(")
        a(f"  alpha = alpha_{scale_name},")
        if config.analysis_level in [AnalysisType.INTERMEDIATE, AnalysisType.ADVANCED]:
            a(f"  omega = omega_{scale_name},")
        a(f"  n_items = length({items_var}),")
        a(f"  sample_size = nrow(na.omit(scale_data))")
        a(")")
        a("")
    
    # Factor analysis
    if opts.include_efa:
        a("# Factor Analysis")
        all_items = [item for scale in scales for item in scale.items]
        a("efa_items <- c(" + ", ".join(f"'{item}'" for item in all_items) + ")")
        a("efa_data <- na.omit(data[efa_items])")
        a("")
        
        if config.parallel_analysis:
            a("# Parallel Analysis for Factor Retention")
            a("pa_result <- psych::fa.parallel(efa_data, fa = 'fa', fm = 'ml')")
            a("n_factors_pa <- pa_result$nfact")
        else:
            a("n_factors_pa <- NULL")
        
        a("")
        a("# Exploratory Factor Analysis")
        a("efa_result <- psych::fa(")
        a("  efa_data,")
        a("  nfactors = n_factors_pa,")
        a("  rotate = 'oblimin',")
        a("  fm = 'ml'")
        a(")")
        a("")
        
        if config.include_plots:
            a("# Factor Analysis Plots")
            a("pdf('factor_analysis_plots.pdf', width = 10, height = 8)")
            a("plot(pa_result)")
            a("psych::fa.diagram(efa_result)")
            a("dev.off()")
            a("")
    
    # Descriptive statistics and correlations
    a("# Descriptive Statistics")
    a("descriptives <- psych::describe(data)")
    a("scale_descriptives <- list()")
    a("")
    
    for scale in scales:
        scale_name = _sanitize_r_name(scale.name)
        a(f"# {scale.name}")
        a(f"scale_data <- data[items_{scale_name}]")
        a(f"scale_score <- rowMeans(scale_data, na.rm = TRUE)")
        a(f"scale_descriptives${scale_name} <- list(")
        a("  items = psych::describe(scale_data),")
        a("  scale_score = psych::describe(scale_score),")
        a("  correlation_matrix = cor(scale_data, use = 'pairwise.complete.obs')")
        a(")")
        a("")
    
    # Correlation matrix
    cor_method = opts.correlation_type
    a("# Correlation Analysis")
    if cor_method == "polychoric":
        a("correlations <- psych::polychoric(data[efa_items])$rho")
    else:
        a(f"correlations <- cor(data[efa_items], method = '{cor_method}', use = 'pairwise.complete.obs')")
    a("")
    
    # Assumption checks
    if config.include_assumptions_checks:
        a("# Statistical Assumptions Checks")
        a("assumption_checks <- list()")
        a("")
        
        for scale in scales:
            scale_name = _sanitize_r_name(scale.name)
            a(f"# {scale.name} assumptions")
            a(f"scale_score <- rowMeans(data[items_{scale_name}], na.rm = TRUE)")
            a(f"assumption_checks${scale_name} <- list(")
            a("  normality = shapiro.test(scale_score)$p.value,")
            a("  n_complete = sum(complete.cases(data[items_" + scale_name + "])),")
            a("  kmo = psych::KMO(data[items_" + scale_name + "])$MSA")
            a(")")
            a("")
    
    # CFA (if requested)
    if config.cfa_model and config.analysis_level == AnalysisType.ADVANCED:
        a("# Confirmatory Factor Analysis")
        a(_generate_cfa_syntax(scales))
        a("")
    
    # Results summary and export
    a("# Results Summary and Export")
    a("analysis_summary <- list(")
    a("  scales = reliability_results,")
    a("  descriptives = scale_descriptives,")
    a("  correlations = correlations")
    
    if opts.include_efa:
        a("  , efa = efa_result")
    if config.include_assumptions_checks:
        a("  , assumptions = assumption_checks")
    if reverse_items:
        a("  , reverse_items = reverse_log")
    
    a(")")
    a("")
    
    # Export options
    a("# Data Export")
    a("haven::write_sav(data, 'processed_data.sav')")
    a("save(analysis_summary, file = 'analysis_results.RData')")
    a("")
    
    # Summary report
    a("# Analysis Summary Report")
    a("cat('\\n=== ANALYSIS SUMMARY ===\\n')")
    a("cat('Scales analyzed:', length(reliability_results), '\\n')")
    a("cat('Total items:', length(efa_items), '\\n')")
    if reverse_items:
        a("cat('Reverse-scored items:', length(reverse_items), '\\n')")
    a("cat('Sample size:', nrow(data), '\\n')")
    a("")
    
    a("# Display reliability coefficients")
    a("for(scale_name in names(reliability_results)) {")
    a("  alpha_val <- reliability_results[[scale_name]]$alpha$total$raw_alpha")
    a("  cat(sprintf('%-20s: α = %.3f\\n', scale_name, alpha_val))")
    a("}")
    a("")
    
    return "\n".join(lines)

def _get_required_packages(config: RScriptConfig, opts: Options) -> List[str]:
    """Determine required R packages based on configuration"""
    packages = ["haven", "psych"]
    
    if opts.include_efa or config.parallel_analysis:
        packages.extend(["GPArotation"])
    
    if config.bootstrap_ci:
        packages.extend(["boot"])
    
    if config.cfa_model:
        packages.extend(["lavaan"])
    
    if config.include_plots:
        packages.extend(["ggplot2", "corrplot"])
    
    return list(set(packages))  # Remove duplicates

def _sanitize_r_name(name: str) -> str:
    """Convert scale name to valid R variable name"""
    import re
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
    # Ensure starts with letter
    if sanitized[0].isdigit():
        sanitized = 'scale_' + sanitized
    return sanitized

def _generate_cfa_syntax(scales: List[Scale]) -> str:
    """Generate lavaan CFA model syntax"""
    model_lines = []
    model_lines.append("# CFA Model Specification")
    model_lines.append("cfa_model <- '")
    
    for scale in scales:
        scale_name = _sanitize_r_name(scale.name)
        items_str = " + ".join(scale.items)
        model_lines.append(f"  {scale_name} =~ {items_str}")
    
    model_lines.append("'")
    model_lines.append("")
    model_lines.append("# Fit CFA Model")
    model_lines.append("cfa_fit <- lavaan::cfa(cfa_model, data = data, std.lv = TRUE)")
    model_lines.append("cfa_summary <- lavaan::summary(cfa_fit, fit.measures = TRUE)")
    model_lines.append("cfa_reliability <- lavaan::reliability(cfa_fit)")
    
    return "\n".join(model_lines)
```

### 4. Improved Data Handling and Validation
**Current Issue**: Basic data cleaning with limited validation.

**Solution - Comprehensive Data Validation**:
```python
from typing import List, Dict, Tuple, Optional
import warnings

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    total_observations: int
    total_variables: int
    missing_patterns: Dict[str, float]
    outliers_detected: Dict[str, int]
    multicollinearity_issues: List[str]
    sample_adequacy: Dict[str, float]
    recommendations: List[str]

def comprehensive_data_validation(df: pd.DataFrame, scales: List[Scale]) -> DataQualityReport:
    """
    Perform comprehensive data quality assessment
    """
    pd = _ensure_pandas()
    report = DataQualityReport(
        total_observations=len(df),
        total_variables=len(df.columns),
        missing_patterns={},
        outliers_detected={},
        multicollinearity_issues=[],
        sample_adequacy={},
        recommendations=[]
    )
    
    # Missing data analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    report.missing_patterns = missing_pct.to_dict()
    
    # Outlier detection using IQR method
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        if outliers > 0:
            report.outliers_detected[col] = outliers
    
    # Multicollinearity check
    for scale in scales:
        if len(scale.items) >= 3:
            try:
                scale_data = df[scale.items].dropna()
                if len(scale_data) > 0:
                    corr_matrix = scale_data.corr()
                    # Check for very high correlations (> 0.9)
                    high_corr = np.where((corr_matrix > 0.9) & (corr_matrix < 1.0))
                    if len(high_corr[0]) > 0:
                        for i, j in zip(high_corr[0], high_corr[1]):
                            if i < j:  # Avoid duplicates
                                item1, item2 = corr_matrix.index[i], corr_matrix.columns[j]
                                report.multicollinearity_issues.append(
                                    f"{item1} - {item2} (r = {corr_matrix.iloc[i,j]:.3f})"
                                )
            except Exception as e:
                logging.warning(f"Could not check multicollinearity for {scale.name}: {e}")
    
    # Sample adequacy (Kaiser-Meyer-Olkin test approximation)
    for scale in scales:
        if len(scale.items) >= 3:
            try:
                scale_data = df[scale.items].dropna()
                if len(scale_data) >= 50:  # Minimum for KMO
                    # Simple adequacy check based on sample size to variables ratio
                    adequacy = len(scale_data) / len(scale.items)
                    report.sample_adequacy[scale.name] = adequacy
            except Exception as e:
                logging.warning(f"Could not assess adequacy for {scale.name}: {e}")
    
    # Generate recommendations
    _generate_recommendations(report)
    
    return report

def _generate_recommendations(report: DataQualityReport) -> None:
    """Generate data quality recommendations"""
    recommendations = []
    
    # Missing data recommendations
    high_missing = {k: v for k, v in report.missing_patterns.items() if v > 10}
    if high_missing:
        recommendations.append(
            f"High missing data detected in {len(high_missing)} variables (>10%). "
            "Consider multiple imputation or listwise deletion."
        )
    
    # Sample size recommendations
    if report.total_observations < 100:
        recommendations.append(
            "Small sample size (<100). Results may be unstable. "
            "Consider collecting more data or using bootstrap methods."
        )
    
    # Outlier recommendations
    if report.outliers_detected:
        total_outliers = sum(report.outliers_detected.values())
        recommendations.append(
            f"Detected {total_outliers} potential outliers across {len(report.outliers_detected)} variables. "
            "Consider winsorizing or robust statistical methods."
        )
    
    # Multicollinearity recommendations
    if report.multicollinearity_issues:
        recommendations.append(
            f"High correlations detected between {len(report.multicollinearity_issues)} item pairs. "
            "Consider removing redundant items or using factor analysis."
        )
    
    # Adequacy recommendations
    low_adequacy = {k: v for k, v in report.sample_adequacy.items() if v < 5}
    if low_adequacy:
        recommendations.append(
            f"Low sample-to-variable ratios in {len(low_adequacy)} scales. "
            "Consider reducing items or collecting more data."
        )
    
    report.recommendations = recommendations

def enhanced_sanitize_metadata(df: pd.DataFrame, meta: Dict) -> Tuple[pd.DataFrame, Dict, DataQualityReport]:
    """
    Enhanced data cleaning with comprehensive quality assessment
    """
    pd = _ensure_pandas()
    
    # Original cleaning logic
    drop_cols: List[str] = []
    for col in df.columns:
        # Enhanced dropping criteria
        nunique = df[col].nunique()
        missing_pct = df[col].isnull().mean()
        
        # Drop if too many categories or mostly missing
        if (df[col].dtype == "object" and nunique > 50) or missing_pct > 0.95:
            drop_cols.append(col)
        # Drop ID-like columns
        elif col.lower() in {"id", "participant_id", "timestamp", "date", "record_id"}:
            drop_cols.append(col)
        # Drop constants
        elif nunique <= 1:
            drop_cols.append(col)
    
    clean_df = df.drop(columns=drop_cols)
    
    # Apply missing value ranges
    for col, ranges in meta.get("missing_ranges", {}).items():
        if col not in clean_df.columns:
            continue
        for lo, hi in ranges:
            clean_df.loc[clean_df[col].between(lo, hi), col] = pd.NA
    
    # Update metadata
    clean_meta = meta.copy()
    clean_meta["dropped"] = drop_cols
    for key in ("var_labels", "var_types", "missing_ranges"):
        if key in clean_meta and isinstance(clean_meta[key], dict):
            clean_meta[key] = {k: v for k, v in clean_meta[key].items() if k not in drop_cols}
    clean_meta["var_names"] = [n for n in meta.get("var_names", []) if n not in drop_cols]
    
    # Generate quality report (basic version for now)
    quality_report = DataQualityReport(
        total_observations=len(clean_df),
        total_variables=len(clean_df.columns),
        missing_patterns={(col: clean_df[col].isnull().mean() * 100) for col in clean_df.columns},
        outliers_detected={},
        multicollinearity_issues=[],
        sample_adequacy={},
        recommendations=[]
    )
    
    return clean_df, clean_meta, quality_report
```

### 5. Logging and Monitoring System
**Current Issue**: Minimal logging functionality.

**Solution - Comprehensive Logging**:
```python
import logging
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import traceback

class SurveyAnalysisLogger:
    """Comprehensive logging system for the Survey-to-R Agent"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("survey_agent")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = self.log_dir / f"survey_agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.session_log = []
    
    def log_session_start(self, file_info: Dict[str, Any]) -> None:
        """Log start of analysis session"""
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "session_start",
            "file_info": file_info
        }
        self.session_log.append(session_data)
        self.logger.info(f"Session {self.session_id} started with file: {file_info.get('name', 'unknown')}")
    
    def log_ai_request(self, variables_count: int, prompt_config: Dict) -> None:
        """Log AI scale detection request"""
        ai_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "ai_request",
            "variables_count": variables_count,
            "prompt_config": prompt_config
        }
        self.session_log.append(ai_data)
        self.logger.info(f"AI scale detection requested for {variables_count} variables")
    
    def log_ai_response(self, scales_detected: int, confidence_scores: List[float]) -> None:
        """Log AI response"""
        ai_response = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "ai_response",
            "scales_detected": scales_detected,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "confidence_distribution": confidence_scores
        }
        self.session_log.append(ai_response)
        self.logger.info(f"AI detected {scales_detected} scales with avg confidence {ai_response['avg_confidence']:.2f}")
    
    def log_user_modifications(self, original_scales: int, final_scales: int, modifications: List[str]) -> None:
        """Log user modifications to AI suggestions"""
        mod_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "user_modifications",
            "original_scales": original_scales,
            "final_scales": final_scales,
            "modifications": modifications
        }
        self.session_log.append(mod_data)
        self.logger.info(f"User modified {len(modifications)} scales: {original_scales} → {final_scales}")
    
    def log_reverse_detection(self, reverse_items: Dict[str, Dict]) -> None:
        """Log reverse item detection results"""
        reverse_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "reverse_detection",
            "items_analyzed": len(reverse_items),
            "items_reversed": sum(1 for info in reverse_items.values() if info.get('is_reverse', False)),
            "avg_confidence": sum(info.get('confidence', 0) for info in reverse_items.values()) / len(reverse_items)
        }
        self.session_log.append(reverse_data)
        self.logger.info(f"Reverse detection: {reverse_data['items_reversed']}/{reverse_data['items_analyzed']} items flagged")
    
    def log_script_generation(self, script_length: int, analysis_options: Dict) -> None:
        """Log R script generation"""
        script_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "script_generation",
            "script_length": script_length,
            "analysis_options": analysis_options
        }
        self.session_log.append(script_data)
        self.logger.info(f"R script generated: {script_length} characters with options {analysis_options}")
    
    def log_error(self, error: Exception, context: str) -> None:
        """Log errors with context"""
        error_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        self.session_log.append(error_data)
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def export_session_log(self) -> str:
        """Export complete session log as JSON"""
        log_file = self.log_dir / f"session_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.session_log, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Session log exported to {log_file}")
        return str(log_file)

# Global logger instance
_logger = None

def get_logger() -> SurveyAnalysisLogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = SurveyAnalysisLogger()
    return _logger

# Update existing functions to use logging
def load_sav_with_logging(path_or_file) -> Tuple[Any, Dict]:
    """Enhanced load_sav with comprehensive logging"""
    logger = get_logger()
    
    try:
        # Determine file info for logging
        if isinstance(path_or_file, (str, os.PathLike)):
            file_info = {"name": str(path_or_file), "type": "path", "size": os.path.getsize(path_or_file)}
        else:
            file_info = {
                "name": getattr(path_or_file, "name", "uploaded_file.sav"),
                "type": "upload",
                "size": len(path_or_file.getvalue()) if hasattr(path_or_file, "getvalue") else "unknown"
            }
        
        logger.log_session_start(file_info)
        
        # Call original function
        result = load_sav(path_or_file)
        
        logger.logger.info(f"Successfully loaded SPSS file with {len(result[0])} observations")
        return result
        
    except Exception as e:
        logger.log_error(e, "load_sav")
        raise
```

## Implementation Priority

### Critical (Must Implement)
1. Real Gemini AI Integration
2. Enhanced error handling and logging
3. Comprehensive reverse item detection

### High Priority  
4. Enhanced R script generation
5. Data quality validation
6. Statistical assumption checks

### Medium Priority
7. Advanced reliability analysis
8. CFA model generation
9. Performance optimizations

### Nice to Have
10. Bootstrap confidence intervals
11. Advanced plotting options
12. Multiple imputation support

These improvements will transform the basic pipeline into a production-ready, scientifically rigorous tool for psychological research.