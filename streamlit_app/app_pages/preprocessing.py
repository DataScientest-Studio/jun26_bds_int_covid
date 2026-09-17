import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "03 · Preprocessing",
    "Every transformation answers an observed data problem",
    "The pipeline is reproducible, conservative, and keeps experimental alternatives explicit.",
)

narrative(
    "Prepare model-ready data without erasing pathology or leaking evaluation information.",
    "Deduplicated first, standardized encoding, aligned masks, stratified splits, and tested lung-focused variants.",
    "21,106 clean images remained, with reproducible 70/15/15 train-validation-test partitions.",
)

st.markdown("#### End-to-end pipeline")
with st.container(horizontal=True, horizontal_alignment="distribute"):
    for label in [
        "21,165 raw",
        "Remove 59 duplicates",
        "Grayscale + resize",
        "Stratified split",
        "Model input",
    ]:
        st.badge(label, color="blue")

st.space("small")

left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
with left:
    figure(
        FIGURES_DIR / "preprocessing_transform_steps.png",
        "Representative transformations from raw X-ray to model-ready input.",
    )
with right:
    st.markdown("#### Controlled choices")
    st.markdown(
        "- **Always:** duplicate removal, grayscale conversion, resizing, deterministic split.\n"
        "- **Compared experimentally:** full image, lung ROI, lungs-only mask, and background-only regions.\n"
        "- **Augmentation:** small rotations; horizontal flipping was removed after it reduced performance.\n"
        "- **Reproducibility:** seed 42, saved manifests, versioned metrics, and tests."
    )
    st.info(
        "Masks were resized with nearest-neighbor interpolation to preserve binary lung boundaries.",
        icon=":material/check_circle:",
    )

footer("Preprocessing")

