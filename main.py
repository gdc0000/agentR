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
        "Carica un file **SPSS (.sav)**, lascia che l’agente proponga i costrutti e genera la sintassi R pronta per l’analisi."
    )

    # ---------------- Sidebar: analysis options ---------------- #
    st.sidebar.header("Opzioni analisi")
    include_efa: bool = st.sidebar.checkbox("Includi EFA", value=True)
    missing_strategy: str = st.sidebar.selectbox(
        "Strategia missing", ["listwise", "pairwise", "mean_scale"]
    )
    correlation_type: str = st.sidebar.selectbox(
        "Tipo di correlazione", ["pearson", "spearman", "polychoric"]
    )
    reverse_threshold: float = st.sidebar.slider(
        "Δα minimo per invertire item", 0.0, 0.2, 0.05, 0.01
    )

    # ---------------- File uploader ---------------- #
    uploaded_file = st.file_uploader("Carica file .sav", type=["sav"])
    if uploaded_file is None:
        st.info("Attendi il caricamento di un file .sav.")
        return

    with st.spinner("Lettura file…"):
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
        with st.spinner("Gemini sta proponendo i costrutti…"):
            scales_prop: List[Scale] = gemini_detect_scales(var_view, prompt_cfg)
            st.session_state.scales_prop = scales_prop
    else:
        scales_prop = st.session_state.scales_prop  # type: ignore

    # ---------------- UI: confirm scales ---------------- #
    st.subheader("Costrutti proposti")
    confirmed_scales: List[Scale] = []

    for i, scale in enumerate(scales_prop):
        with st.expander(f"Costrutto {i + 1}: {scale.name} ({len(scale.items)} item)"):
            new_name = st.text_input("Nome costrutto", scale.name, key=f"name_{i}")
            keep = st.checkbox(
                "Mantieni questo costrutto", value=True, key=f"keep_{i}"
            )
            selected_items = st.multiselect(
                "Item inclusi",
                options=var_view_df["name"].tolist(),
                default=scale.items,
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

    st.markdown(f"**Costrutti confermati:** {len(confirmed_scales)}")

    # ---------------- Generate R script ---------------- #
    if st.button("Genera sintassi R", disabled=len(confirmed_scales) == 0):
        opts = Options(
            include_efa=include_efa,
            missing_strategy=missing_strategy,
            correlation_type=correlation_type,
            reverse_threshold=reverse_threshold,
        )

        with st.spinner("Determinazione item inversi…"):
            rev_map = detect_reverse_items(confirmed_scales, clean_df)

        with st.spinner("Costruzione sintassi R…"):
            r_script: str = build_r_syntax(
                sav_path=uploaded_file.name,
                scales=confirmed_scales,
                rev_map=rev_map,
                opts=opts,
            )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"analysis_{timestamp}.R"

        st.download_button(
            label="Scarica sintassi R",
            data=r_script,
            file_name=filename,
            mime="text/plain",
        )
        st.success("Sintassi R generata!")


if __name__ == "__main__":
    main()
