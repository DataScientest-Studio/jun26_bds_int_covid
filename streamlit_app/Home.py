import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from lib.ui import app_footer, configure_page, cover_hero, disclaimer_banner, mini_steps

configure_page("Home")

cover_hero(
    eyebrow="Data Science Bootcamp &middot; Final Project Defense",
    title="COVID-19 Chest X-Ray Classification",
    authors="Berfin &nbsp;&amp;&nbsp; Mert",
    context=(
        "Telling COVID-19 chest X-rays apart from Normal, Viral Pneumonia, and Lung "
        "Opacity cases with a full deep learning pipeline, from raw data to a live demo."
    ),
)

disclaimer_banner()

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Chest X-rays", "21,165")
metric_cols[1].metric("Diagnostic classes", "4")
metric_cols[2].metric("Models trained", "13+")
metric_cols[3].metric("Best test accuracy", "93.78%")

st.markdown("#### Pipeline")
mini_steps(
    [
        "Dataset & EDA",
        "Preprocessing",
        "Classical & Baseline Models",
        "Deep CNN From Scratch",
        "Transfer Learning",
        "Interpretability & Limitations",
        "Conclusion",
        "Live Demo",
    ]
)

st.caption("Use the sidebar to move through the pipeline step by step.")

app_footer()
