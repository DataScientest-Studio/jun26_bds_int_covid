import streamlit as st

from lib.ui import disclaimer, footer, page_intro


page_intro(
    "06 · Conclusion",
    "The score is strong; the claim must stay narrow",
    "The project succeeds as a reproducible classifier and as an audit of why internal performance can mislead.",
)

columns = st.columns(3, gap="medium")
sections = [
    (
        ":material/check_circle:",
        "What worked",
        "A strong, reproducible four-class classifier with a fixed split, saved artifacts, class-sensitive evaluation, and an interactive inference path.",
    ),
    (
        ":material/warning:",
        "Main limitation",
        "Disease labels and acquisition sources are entangled. Internal accuracy therefore mixes pathology signal with dataset-specific shortcuts.",
    ),
    (
        ":material/next_plan:",
        "What comes next",
        "Validate on separated sources first; only then optimize architectures, robustness, calibration, and clinical workflow.",
    ),
]
for column, (icon, title, body) in zip(columns, sections):
    with column.container(border=True, height="stretch"):
        st.markdown(f"### {icon} {title}")
        st.write(body)

st.markdown("#### Future work—not completed claims")
future = [
    ("Source-separated external validation", "Test whether performance survives new hospitals, scanners, and acquisition pipelines."),
    ("Domain harmonization or adversarial training", "Reduce source-identifying signal without erasing disease-relevant structure."),
    ("Better lung-constrained architectures", "Constrain features spatially while preserving the context needed for classification."),
    ("Calibration and prospective clinical evaluation", "Measure probability reliability and workflow value under clinical oversight."),
]
for title, detail in future:
    with st.container(border=True):
        st.markdown(f"**{title}**  ")
        st.caption(detail)

st.success(
    "Final takeaway: model performance is an engineering result; trustworthiness is a separate scientific question.",
    icon=":material/flag:",
)
disclaimer()
footer("Conclusion")
