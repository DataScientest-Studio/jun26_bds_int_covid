import streamlit as st

from lib.paths import REPORT_PDF
from lib.ui import disclaimer, footer


st.caption("DATA SCIENCE BOOTCAMP · FINAL PROJECT")
st.title("COVID-19 chest X-ray classification")
st.markdown("### Four-class classification of lung X-ray images")
st.write(
    "We study chest X-rays from the COVID-19 Radiography Database and ask whether a "
    "high-performing image classifier learns from lung tissue or from dataset-specific shortcuts."
)

with st.container(horizontal=True, gap="small"):
    st.badge("21,165 chest X-rays", icon=":material/radiology:", color="blue")
    st.badge("4 lung-image classes", icon=":material/category:", color="violet")
    st.badge("92.4% raw-score winner", icon=":material/analytics:", color="green")
    st.badge("Shortcut learning identified", icon=":material/warning:", color="orange")

st.space("medium")

with st.container(border=True):
    st.subheader("Project goal")
    st.write(
        "Train and evaluate models that separate COVID, Lung Opacity, Normal, and Viral "
        "Pneumonia chest X-rays, then test whether strong internal accuracy reflects "
        "medically relevant lung evidence."
    )
    st.markdown(
        "**Research question:** Can a high-performing classifier be trusted to learn from "
        "the lungs—or from how the dataset was assembled?"
    )

with st.container(horizontal=True, gap="large"):
    st.markdown("**Authors:** Berfin & Mert")
    st.markdown("**Project mentor:** Paul Grolier")
    st.markdown("**Format:** 20-minute walkthrough + live prediction")

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
