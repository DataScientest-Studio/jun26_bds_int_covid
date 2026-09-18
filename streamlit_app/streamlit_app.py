import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.ui import configure_page

configure_page()

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
    st.Page("app_pages/data_audit.py", title="Data audit", icon=":material/fact_check:"),
    st.Page("app_pages/preprocessing.py", title="Preprocessing", icon=":material/tune:"),
    st.Page("app_pages/modelling.py", title="Modelling", icon=":material/model_training:"),
    st.Page("app_pages/results.py", title="Results", icon=":material/leaderboard:"),
    st.Page("app_pages/trust.py", title="Trust & limits", icon=":material/policy:"),
    st.Page("app_pages/conclusion.py", title="Conclusion", icon=":material/flag:"),
    st.Page("app_pages/live_demo.py", title="Live demo", icon=":material/upload_file:"),
]

navigation = st.navigation(pages, position="top")
navigation.run()
