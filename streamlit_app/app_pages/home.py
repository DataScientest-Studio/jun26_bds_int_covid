import streamlit as st

from lib.paths import REPORT_PDF
from lib.ui import disclaimer, footer


st.caption("DATA SCIENCE BOOTCAMP · FINAL PROJECT")
st.title("COVID-19 chest X-ray classification")
st.markdown("### Four-class classification of chest radiographs")
st.write(
    "We classify chest X-rays from the COVID-19 Radiography Database and ask what a "
    "high-performing classifier actually uses: the lungs, or characteristics of the "
    "repositories the images were collected from."
)

with st.container(horizontal=True, gap="small"):
    st.badge("21,106 X-rays after deduplication", icon=":material/radiology:", color="blue")
    st.badge("4 diagnostic classes", icon=":material/category:", color="violet")
    st.badge("Best model: 0.936 macro F1", icon=":material/analytics:", color="green")
    st.badge("Shortcut learning identified", icon=":material/warning:", color="orange")

st.space("medium")

with st.container(border=True):
    st.subheader("Two research questions")
    st.markdown(
        "**1. Performance.** How accurately can convolutional networks classify "
        "COVID-19, Lung Opacity, Normal, and Viral Pneumonia — and how much does "
        "transfer learning improve on a network trained from scratch?"
    )
    st.markdown(
        "**2. Attribution.** Which parts of the image do these models rely on, and "
        "how much of their accuracy comes from the lungs rather than from the source "
        "repositories?"
    )

with st.container(horizontal=True, gap="large"):
    st.markdown("**Authors:** Berfin Aktaş & Mert Doğruca")
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