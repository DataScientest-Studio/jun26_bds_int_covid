import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from lib.ui import configure

configure()

pages = {
    "Presentation": [
        st.Page("app_pages/00_opening.py", title="Opening", icon=":material/radiology:", default=True),
        st.Page("app_pages/01_question.py", title="01 · The question", icon=":material/help_outline:"),
        st.Page("app_pages/02_dataset.py", title="02 · The material", icon=":material/database:"),
        st.Page("app_pages/03_audit.py", title="03 · The audit", icon=":material/search:"),
        st.Page("app_pages/04_pipeline.py", title="04 · The controls", icon=":material/account_tree:"),
        st.Page("app_pages/05_baselines.py", title="05 · The floor", icon=":material/straighten:"),
        st.Page("app_pages/06_cnn.py", title="06 · From scratch", icon=":material/memory:"),
        st.Page("app_pages/07_transfer.py", title="07 · Borrowed vision", icon=":material/model_training:"),
        st.Page("app_pages/08_suspicion.py", title="08 · The suspicion", icon=":material/visibility:"),
        st.Page("app_pages/09_causal_test.py", title="09 · The causal test", icon=":material/science:"),
        st.Page("app_pages/10_decomposition.py", title="10 · What survives", icon=":material/layers:"),
        st.Page("app_pages/11_limits.py", title="11 · The boundary", icon=":material/warning:"),
        st.Page("app_pages/12_verdict.py", title="12 · The verdict", icon=":material/fact_check:"),
    ],
    "Try it": [
        st.Page("app_pages/13_live_classifier.py", title="13 · Live classifier", icon=":material/play_circle:"),
    ],
}

st.navigation(pages, position="sidebar", expanded=False).run()

