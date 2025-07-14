# app.py
# Streamlit web app: SocialSurveyAgent
# Requirements (pip install):
#   streamlit pyreadstat pandas numpy scipy scikit-learn pingouin factor_analyzer google-generativeai

import os, io, time, json, tempfile, datetime
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import pingouin as pg
from sklearn.cluster import AgglomerativeClustering
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import google.generativeai as genai

# ───────────────────────────────────────────────────────────────
#  1. Gemini rate-limiter (free-tier 15 RPM / 50 RPD → safe margin)
# ───────────────────────────────────────────────────────────────
class GeminiRateLimiter:
    def __init__(self, max_rpm=12, max_rpd=45):
        self.max_rpm, self.max_rpd = max_rpm, max_rpd
        self.calls_minute = 0
        self.calls_day = 0
        self.minute_start = time.time()
        self.day_start = datetime.date.today()

    def _tick(self):
        now = time.time()
        if now - self.minute_start >= 60:
            self.calls_minute = 0
            self.minute_start = now
        if datetime.date.today() != self.day_start:
            self.calls_day = 0
            self.day_start = datetime.date.today()

    def wait(self):
        self._tick()
        if self.calls_day >= self.max_rpd:
            st.error("Gemini daily quota reached. Try tomorrow."); st.stop()
        while self.calls_minute >= self.max_rpm:
            self._tick()
            time.sleep(1)
        self.calls_minute += 1
        self.calls_day += 1

rate_limiter = GeminiRateLimiter()

# ───────────────────────────────────────────────────────────────
#  2. Utility functions
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_sav(uploaded_file):
    """
    Salva l’UploadedFile Streamlit in un file temporaneo
    e lo passa a pyreadstat.read_sav().
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()           # assicura che tutto sia scritto

        # pyreadstat legge dal path
        df, meta = pyreadstat.read_sav(tmp.name, apply_value_formats=False)

    return df, meta

def labels_map(meta) -> dict:
    """
    Ritorna dict {var_name: var_label}.
    Gestisce meta.column_labels che è una lista parallela a meta.column_names.
    """
    return dict(zip(meta.column_names, meta.column_labels))

def variable_view(df, meta):
    lb = labels_map(meta)           # NEW
    types_map = getattr(meta, "variable_types", {})
    value_fmt = getattr(meta, "variable_value_formats", {})

    rows = []
    for col in df.columns:
        rows.append({
            "var_name":  col,
            "label":     lb.get(col, ""),      # UPDATED
            "type":      types_map.get(col, ""),
            "%missing":  df[col].isna().mean().round(3),
            "min":       df[col].min(skipna=True),
            "max":       df[col].max(skipna=True),
            "value_fmt": value_fmt.get(col, "")
        })
    return pd.DataFrame(rows)

def gemini_embed(texts, model):
    rate_limiter.wait()
    return model.embed_content(texts=texts)["embedding"]

def gemini_group_items(texts, model):
    rate_limiter.wait()
    prompt = (
        "Group the following survey items into psychological constructs.\n"
        "Return JSON mapping construct_name to list of item_names.\n\nItems:\n"
        + "\n".join(texts)
    )
    rsp = model.generate_content(prompt)
    try:
        return json.loads(rsp.text)
    except Exception:
        return {}

def detect_constructs(labels, model):
    # Embedding clustering
    emb = np.vstack(gemini_embed(labels, model))
    if len(labels) > 1:
        clusters = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.3
        ).fit_predict(emb)
    else:
        clusters = np.zeros(1, dtype=int)
    draft = {}
    for idx, lab in enumerate(labels):
        draft.setdefault(f"Cluster{clusters[idx]+1}", []).append(lab)
    # Gemini refinement
    llm_map = gemini_group_items(labels, model)
    if llm_map:
        draft.update(llm_map)
    return draft

def cronbach_alpha(df):
    return pg.cronbach_alpha(data=df)[0]

def mcdonald_omega(df):
    # quick one-factor FA omega_t
    fa = FactorAnalyzer(n_factors=1, method="minres")
    fa.fit(df.dropna())
    load = fa.loadings_.flatten()
    uniq = fa.get_uniquenesses()
    num = (load.sum())**2
    den = num + uniq.sum()
    return num / den

def reverse_if_improves(df_items):
    reversed_cols = []
    df = df_items.copy()
    for col in df.columns:
        if any(k in col.lower() for k in ("not", "reverse", "_r", "neg")):
            df_rev = df.copy()
            df_rev[col] = df_rev[col].max() + df_rev[col].min() - df_rev[col]
            if cronbach_alpha(df_rev) > cronbach_alpha(df):
                df = df_rev
                reversed_cols.append(col)
    return df, reversed_cols

def efa_summary(df_scale, n_factors):
    fa = FactorAnalyzer(n_factors=n_factors, rotation="promax")
    fa.fit(df_scale.dropna())
    loadings = pd.DataFrame(fa.loadings_, index=df_scale.columns)
    return fa, loadings

def write_enhanced_sav(df, meta, path):
    pyreadstat.write_sav(df, path, variable_value_labels=meta.variable_value_labels,
                         variable_labels=meta.column_labels)

def generate_r_script(constructs, reversed_items, miss_method, efa_params, outfile):
    ts = datetime.date.today().isoformat()
    lines = [
        "# Auto-generated by SocialSurveyAgent",
        f"# Date: {ts}\n",
        "library(tidyverse)",
        "library(haven)",
        "library(psych)",
        "library(GPArotation)",
        "library(MBESS)",
        "library(lavaan)",
        "library(mice)\n",
        "dat <- read_sav('enhanced_dataset.sav')\n"
    ]
    for c_name, items in constructs.items():
        sc = f"{c_name}_SCORE"
        items_vec = ", ".join([f"'{i}'" for i in items])
        lines.append(f"{sc} <- rowMeans(select(dat, {items_vec}), na.rm = TRUE)")
    if reversed_items:
        ri = ", ".join([f"'{i}'" for i in reversed_items])
        lines.append(f"# Reversed items: {ri}")
    lines.append(f"# Missing data method: {miss_method}")
    if efa_params:
        lines.append("# EFA parameters stored separately")
    with open(outfile, "w") as f:
        f.write("\n".join(lines))

# ───────────────────────────────────────────────────────────────
#  3. Streamlit UI
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SocialSurveyAgent", layout="wide")
st.title("SocialSurveyAgent")

api_key = st.text_input("Gemini API Key", type="password", help="Required (Google Generative AI)")
uploaded = st.file_uploader("Upload .sav file", type=["sav"])

if api_key and uploaded:
    os.environ["GEMINI_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest", generation_config={"temperature": 0})
    st.success("Gemini configured.")

    df, meta = read_sav(uploaded)
    st.subheader("Variable view")
    st.dataframe(variable_view(df, meta), use_container_width=True)

    # Construct detection
    if st.button("Detect constructs"):
        lb = labels_map(meta)                            # NEW
        labels = [lb.get(c, c) for c in df.columns]      # UPDATED
        constructs_raw = detect_constructs(labels, gemini_model)
        st.session_state.constructs = constructs_raw


    if "constructs" in st.session_state:
        st.subheader("Confirm constructs")
        constructs_final = {}
        for cname, items in st.session_state.constructs.items():
            with st.expander(f"{cname} (items: {len(items)})", expanded=False):
                new_name = st.text_input("Construct name", cname, key=f"name_{cname}")
                sel_items = st.multiselect("Items in construct", df.columns.tolist(),
                                           default=[i.split("|")[-1].strip() for i in items],
                                           key=f"items_{cname}")
                if st.checkbox("Keep", value=True, key=f"keep_{cname}"):
                    constructs_final[new_name] = sel_items
        st.session_state.constructs_confirmed = constructs_final

    # Analysis
    if st.session_state.get("constructs_confirmed") and st.button("Run analysis"):
        constructs = st.session_state.constructs_confirmed
        reversed_items_global = []
        scale_scores = {}
        log_lines = []
        for c, items in constructs.items():
            df_items = df[items].astype(float)
            df_items_clean, reversed_cols = reverse_if_improves(df_items)
            reversed_items_global.extend(reversed_cols)
            alpha = cronbach_alpha(df_items_clean)
            omega = mcdonald_omega(df_items_clean)
            scale = df_items_clean.mean(axis=1)
            scale_scores[f"{c}_SCORE"] = scale
            log_lines.append(f"{c}: alpha={alpha:.3f}, omega={omega:.3f}, reversed={reversed_cols}")

        for sc_name, sc_ser in scale_scores.items():
            df[sc_name] = sc_ser

        # Minimal descriptives and correlations
        desc = df[list(scale_scores.keys())].describe().T
        st.subheader("Scale descriptives")
        st.dataframe(desc)

        corr = df[list(scale_scores.keys())].corr(method="pearson")
        st.subheader("Scale correlations")
        st.dataframe(corr)

        # Save enhanced .sav
        out_dir = tempfile.mkdtemp()
        sav_path = os.path.join(out_dir, "enhanced_dataset.sav")
        write_enhanced_sav(df, meta, sav_path)

        # R script
        r_path = os.path.join(out_dir, "analysis.R")
        generate_r_script(constructs, reversed_items_global, "listwise", None, r_path)

        md_path = os.path.join(out_dir, "log.md")
        with open(md_path, "w") as f:
            f.write("# Analysis log\n\n")
            f.write("\n".join(log_lines))

        st.download_button("Download enhanced .sav", data=open(sav_path, "rb").read(), file_name="enhanced_dataset.sav")
        st.download_button("Download R script", data=open(r_path, "rb").read(), file_name="analysis.R")
        st.download_button("Download log", data=open(md_path, "rb").read(), file_name="log.md")

        st.success("Done.")
