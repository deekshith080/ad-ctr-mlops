"""
streamlit_app.py

Visual monitoring dashboard for ad-ctr-mlops.
Shows model health, drift reports, and live predictions.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ad CTR MLOps Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ── Helper functions ───────────────────────────────────────────────────────────

def get_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_model_info():
    try:
        r = requests.get(f"{API_BASE}/model/info", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_prediction(features: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=features, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def load_latest_drift_report():
    reports_dir = Path("reports/drift")
    if not reports_dir.exists():
        return None
    files = sorted(reports_dir.glob("drift_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Ad CTR MLOps Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("Refresh"):
    st.rerun()

st.markdown("---")

# ── Row 1: Model Health ────────────────────────────────────────────────────────
st.subheader("Model Health")

health   = get_api_health()
model_info = get_model_info()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if health and health.get("model_loaded"):
        st.success("API Status: Healthy")
    else:
        st.error("API Status: Unavailable")

with col2:
    if model_info:
        st.metric("Baseline AUC", model_info.get("baseline_auc", "N/A"))
    else:
        st.metric("Baseline AUC", "N/A")

with col3:
    if model_info:
        run_id = model_info.get("run_id", "")[:8]
        st.metric("Model Run ID", run_id + "...")
    else:
        st.metric("Model Run ID", "N/A")

with col4:
    if health:
        st.metric("Model Loaded", "Yes")
    else:
        st.metric("Model Loaded", "No")

st.markdown("---")

# ── Row 2: Drift Report ────────────────────────────────────────────────────────
st.subheader("Latest Drift Report")

report = load_latest_drift_report()

if report:
    summary = report.get("summary", {})
    scores  = report.get("feature_scores", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean PSI", summary.get("mean_psi", 0))
    with col2:
        st.metric("Max PSI", summary.get("max_psi", 0))
    with col3:
        st.metric("Stable Features", summary.get("n_stable", 0))
    with col4:
        retrain = summary.get("retrain_recommended", False)
        if retrain:
            st.error("Retrain Recommended")
        else:
            st.success("No Retrain Needed")

    if scores:
        st.markdown("**PSI Scores by Feature**")
        psi_df = pd.DataFrame([
            {
                "Feature":  feat,
                "PSI":      info["psi"],
                "Type":     info["type"],
                "Severity": info["severity"],
            }
            for feat, info in scores.items()
        ]).sort_values("PSI", ascending=False)

        # Color code by severity
        def color_severity(val):
            if val == "severe_drift":
                return "background-color: #ffcccc"
            elif val == "moderate_drift":
                return "background-color: #fff3cc"
            else:
                return "background-color: #ccffcc"

        st.dataframe(
            psi_df.style.applymap(color_severity, subset=["Severity"]),
            use_container_width=True,
            height=300,
        )

        st.markdown("**PSI Bar Chart**")
        chart_df = psi_df.set_index("Feature")["PSI"]
        st.bar_chart(chart_df)

    st.caption(f"Report generated: {report.get('run_timestamp', 'N/A')}")
else:
    st.info("No drift reports found. Run drift_detector.py first.")

st.markdown("---")

# ── Row 3: Live Prediction ─────────────────────────────────────────────────────
st.subheader("Live CTR Prediction")
st.caption("Enter ad impression features to get a real-time CTR prediction from the API.")

with st.form("prediction_form"):
    st.markdown("**Numeric Features**")
    nc = st.columns(7)
    int_vals = {}
    for i in range(13):
        with nc[i % 7]:
            int_vals[f"intCol_{i}"] = st.number_input(
                f"intCol_{i}", value=1.0, step=1.0
            )

    st.markdown("**Categorical Features**")
    cc = st.columns(4)
    cat_vals = {}
    defaults = [
        "05db9164", "08d6d899", "a99f214a", "5b392875",
        "43b19349", "6f6d9be8", "bcdee96c", "cada4365",
        "001f3601", "07d13a8f", "1f89b562", "a7b606c4",
        "06367733", "None", "None", "32c7478e",
        "3fdb382b", "None", "None", "None",
        "None", "None", "None", "None", "None", "None",
    ]
    for i in range(26):
        with cc[i % 4]:
            cat_vals[f"catCol_{i}"] = st.text_input(
                f"catCol_{i}", value=defaults[i]
            )

    submitted = st.form_submit_button("Get CTR Prediction")

if submitted:
    features = {**int_vals, **cat_vals}
    with st.spinner("Getting prediction from API..."):
        result = get_prediction(features)
    if result:
        col1, col2, col3 = st.columns(3)
        with col1:
            prob = result.get("ctr_probability", 0)
            st.metric("CTR Probability", f"{prob:.2%}")
        with col2:
            will_click = result.get("will_click", False)
            if will_click:
                st.success("Prediction: Will Click")
            else:
                st.warning("Prediction: Will Not Click")
        with col3:
            st.metric("Model Version", result.get("model_version", "")[:8] + "...")
        st.progress(float(result.get("ctr_probability", 0)))
    else:
        st.error("Could not get prediction. Is the API running at localhost:8000?")

st.markdown("---")
st.caption("ad-ctr-mlops | XGBoost + MLflow + Airflow + FastAPI + Streamlit")
