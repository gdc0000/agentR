import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List

from survey_to_r import (
    load_sav,
    sanitize_metadata,
    summarize_variables,
    gemini_detect_scales,
    detect_reverse_items,
    build_r_syntax,
    Options,
    PromptConfig,
    Scale,
    VariableInfo,
)

st.set_page_config(page_title="Survey‑to‑R Agent", layout="wide")


def main() -> None:
    """Streamlit GUI main entry point."""

    st.title("Survey‑to‑R Agent")
    st.markdown(
        "Upload an **SPSS (.sav)** file, let the agent propose constructs, and generate R syntax ready for analysis."
    )

    # ---------------- Sidebar: LLM Configuration ---------------- #
    st.sidebar.header("LLM Configuration")
    llm_provider = st.sidebar.selectbox("Provider", ["Gemini", "OpenRouter"])
    
    llm_api_key = ""
    llm_model = ""
    
    if llm_provider == "OpenRouter":
        llm_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
        llm_model = st.sidebar.text_input("Model Name", value="google/gemini-2.0-flash-001")
    else:
        llm_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Required if not set in GEMINI_API_KEY env var")
        llm_model = "gemini-pro"

    # ---------------- Sidebar: analysis options ---------------- #
    st.sidebar.header("Analysis Options")
    include_efa: bool = st.sidebar.checkbox("Include EFA", value=True)
    missing_strategy: str = st.sidebar.selectbox(
        "Missing Value Strategy", ["listwise", "pairwise", "mean_scale"]
    )
    correlation_type: str = st.sidebar.selectbox(
        "Correlation Type", ["pearson", "spearman", "polychoric"]
    )
    reverse_threshold: float = st.sidebar.slider(
        "Min Δα to reverse item", 0.0, 0.2, 0.05, 0.01
    )

    # ---------------- File uploader ---------------- #
    uploaded_file = st.file_uploader("Upload .sav file", type=["sav"])
    if uploaded_file is None:
        st.info("Please upload a .sav file.")
        return

    with st.spinner("Reading file…"):
        df, meta = load_sav(uploaded_file)
        clean_df, clean_meta = sanitize_metadata(df, meta)
        var_view: List[VariableInfo] = summarize_variables(clean_meta)
        var_view_df = pd.DataFrame(var_view)

    st.subheader("Variable view")
    st.dataframe(var_view_df, use_container_width=True)

    # ---------------- LLM: detect scales ---------------- #
    prompt_cfg = PromptConfig(
        system_prompt="Group survey items into psychological constructs.",
        temperature=0.2,
        top_p=0.9,
    )

    if "scales_prop" not in st.session_state:
        with st.spinner(f"{llm_provider} is proposing constructs…"):
            scales_prop: List[Scale] = gemini_detect_scales(
                var_view, 
                prompt_cfg,
                provider=llm_provider.lower(),
                api_key=llm_api_key,
                model_name=llm_model
            )
            if not scales_prop:
                st.error("No constructs were proposed. Please check your API Key (sidebar) and try again.")
            st.session_state.scales_prop = scales_prop
    else:
        scales_prop = st.session_state.scales_prop  # type: ignore

    # ---------------- UI: confirm scales ---------------- #
    st.subheader("Proposed Constructs")
    confirmed_scales: List[Scale] = []

    for i, scale in enumerate(scales_prop):
        with st.expander(f"Construct {i + 1}: {scale.name} ({len(scale.items)} items)"):
            new_name = st.text_input("Construct Name", scale.name, key=f"name_{i}")
            keep = st.checkbox(
                "Keep this construct", value=True, key=f"keep_{i}"
            )
            available_options = var_view_df["name"].tolist()
            valid_defaults = [item for item in scale.items if item in available_options]
            
            if len(valid_defaults) < len(scale.items):
                missing_items = set(scale.items) - set(valid_defaults)
                st.warning(f"The following items were proposed but not found in the data: {', '.join(missing_items)}")

            selected_items = st.multiselect(
                "Included Items",
                options=available_options,
                default=valid_defaults,
                key=f"items_{i}",
            )
            if keep and selected_items:
                confirmed_scales.append(
                    Scale(
                        name=new_name,
                        items=selected_items,
                        confidence=scale.confidence,
                        note=scale.note,
                    )
                )

    st.markdown(f"**Confirmed Constructs:** {len(confirmed_scales)}")

    # ---------------- Generate R script ---------------- #
    if st.button("Generate R Syntax", disabled=len(confirmed_scales) == 0):
        opts = Options(
            include_efa=include_efa,
            missing_strategy=missing_strategy,
            correlation_type=correlation_type,
            reverse_threshold=reverse_threshold,
        )

        with st.spinner("Detecting reverse items…"):
            rev_map = detect_reverse_items(confirmed_scales, clean_df)

        with st.spinner("Building R syntax…"):
            r_script: str = build_r_syntax(
                sav_path=uploaded_file.name,
                scales=confirmed_scales,
                rev_map=rev_map,
                opts=opts,
            )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"analysis_{timestamp}.R"

        st.download_button(
            label="Download R Syntax",
            data=r_script,
            file_name=filename,
            mime="text/plain",
        )
        st.success("R Syntax generated!")


if __name__ == "__main__":
    main()
