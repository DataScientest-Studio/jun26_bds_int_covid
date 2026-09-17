import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "01 · Dataset and question",
    "A useful dataset with a dangerous structure",
    "Four diagnoses, four source histories, and several ways a model could learn the wrong lesson.",
)

narrative(
    "Classify chest X-rays as COVID, Lung Opacity, Normal, or Viral Pneumonia.",
    "Audited every image, mask, label, encoding, source, and class frequency before modelling.",
    "The data is large enough to model, but class and source are entangled—a direct shortcut-learning risk.",
)

left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
with left:
    figure(
        FIGURES_DIR / "class_distribution.png",
        "Raw class distribution: Normal is almost half of the dataset.",
    )
with right:
    with st.container(horizontal=True):
        st.metric("Raw images", "21,165", border=True)
        st.metric("Image size", "299 × 299", border=True)
    with st.container(horizontal=True):
        st.metric("Normal", "48.2%", border=True)
        st.metric("Viral pneumonia", "6.4%", border=True)
    st.markdown("#### Statistical validation")
    st.write(
        "The class distribution differs strongly from an equal split: χ² = 8,110.78, p < 0.001, with a large effect size (Cohen’s w = 0.619)."
    )
    st.info(
        "Business implication: accuracy alone is insufficient. We prioritize macro F1, per-class recall, and confusion matrices.",
        icon=":material/lightbulb:",
    )

footer("Dataset")

