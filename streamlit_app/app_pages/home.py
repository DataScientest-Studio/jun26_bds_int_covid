import streamlit as st

from lib.paths import REPORT_PDF
from lib.ui import disclaimer, footer


st.caption("DATA SCIENCE BOOTCAMP · FINAL PROJECT")
st.title("COVID-19 chest X-ray classification")
st.markdown(
    "## Can a high-performing image classifier be trusted to learn from the lungs?"
)
st.write(
    "A four-class computer-vision study spanning data exploration, preprocessing, classical machine learning, deep learning, transfer learning, and explainability."
)

with st.container(horizontal=True):
    st.badge("21,165 raw X-rays", icon=":material/image:", color="blue")
    st.badge("4 classes", icon=":material/category:", color="violet")
    st.badge("93.8% best test accuracy", icon=":material/analytics:", color="green")
    st.badge("Shortcut learning identified", icon=":material/warning:", color="orange")

st.space("medium")

left, right = st.columns([1.3, 1], gap="large", vertical_alignment="center")
with left:
    with st.container(border=True):
        st.subheader("The defense in one sentence")
        st.write(
            "We built increasingly capable models, then tested whether their performance came from clinically relevant lung information or from dataset-specific shortcuts."
        )
        st.markdown("**Authors:** Berfin & Mert  ")
        st.markdown("**Project mentor:** Paul Grolier  ")
        st.markdown("**Format:** 20-minute walkthrough + live prediction")
with right:
    st.markdown("#### Presentation route")
    st.markdown(
        "1. Dataset and question\n"
        "2. Exploration and bias checks\n"
        "3. Preprocessing\n"
        "4. Model progression\n"
        "5. Held-out results\n"
        "6. Interpretability and limitations\n"
        "7. Live demo"
    )

if REPORT_PDF.exists():
    with REPORT_PDF.open("rb") as report_file:
        st.download_button(
            "Download the full report",
            data=report_file.read(),
            file_name="covid_xray_project_report.pdf",
            mime="application/pdf",
            icon=":material/download:",
        )

disclaimer()
footer("Opening")

