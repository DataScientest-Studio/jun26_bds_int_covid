import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "02 · Exploration",
    "Exploration was a bias audit, not just a gallery",
    "We turned visual observations into testable risks and preprocessing decisions.",
)

narrative(
    "Technical artifacts can correlate with labels and inflate evaluation scores.",
    "Measured encoding, file size, duplicates, brightness, contrast, and lung-mask geometry by class.",
    "We found class-linked RGB encoding, 59 redundant images, and systematic intensity differences.",
)

tab1, tab2, tab3 = st.tabs(["Encoding", "Duplicates", "Intensity"])
with tab1:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "viral_pneumonia_file_size.png",
            "All 140 RGB images occur in Viral Pneumonia and are larger than grayscale files.",
        )
    with right:
        st.metric("RGB images", "140", "10.4% of Viral Pneumonia", border=True)
        st.write(
            "Encoding is a technical property, not pathology. Converting every image to grayscale prevents this easy shortcut."
        )
with tab2:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "duplicate_distribution.png",
            "Redundant exact duplicates are concentrated in the COVID class.",
        )
    with right:
        st.metric("Redundant files", "59", "0.28% of the dataset", border=True)
        st.write(
            "Removing duplicates before splitting prevents identical observations from crossing into evaluation data."
        )
with tab3:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "brightness_contrast_boxplots.png",
            "Whole-image brightness and contrast differ by class.",
        )
    with right:
        st.metric("COVID mean brightness", "139.5", border=True)
        st.metric("Normal mean contrast", "61.6", border=True)
        st.write(
            "These global signals may reflect acquisition pipelines or backgrounds, motivating lung-focused ablations later."
        )

footer("Exploration")

