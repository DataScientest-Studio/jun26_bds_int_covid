import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import disclaimer, figure, footer, narrative, page_intro


page_intro(
    "06 · Interpretability and limits",
    "A good score is not the same as a trustworthy model",
    "Grad-CAM suggested weak lung focus; a controlled intervention showed why that matters.",
)

narrative(
    "Determine whether predictions depend on lung pathology or on source-specific background cues.",
    "Visualized Grad-CAM, measured attention inside lung masks, then retrained the same model with the background removed.",
    "COVID recall fell from 90.3% to 57.0%—direct evidence of COVID-specific background leakage.",
)

with st.container(horizontal=True):
    st.metric("Full-image COVID recall", "90.3%", border=True)
    st.metric("Lungs-only COVID recall", "57.0%", "−33.3 points", border=True)
    st.metric("Overall accuracy change", "−6.6 points", border=True)

tab1, tab2 = st.tabs(["Visual evidence", "Causal test"])
with tab1:
    left, right = st.columns(2, gap="large")
    with left:
        figure(
            FIGURES_DIR / "gradcam_transfer_baseline.png",
            "Full-image model: activation frequently extends to borders and shoulders.",
        )
    with right:
        figure(
            FIGURES_DIR / "transfer_lung_focus_baseline.png",
            "Attention inside the lungs is only mildly above chance and below chance for Lung Opacity.",
        )
with tab2:
    left, right = st.columns(2, gap="large")
    with left:
        figure(
            FIGURES_DIR / "gradcam_transfer_masked.png",
            "After masking the background, the model is forced to rely on lung content.",
        )
    with right:
        st.subheader("What this establishes")
        st.write(
            "Masking changes one causal input: access to non-lung pixels. The disproportionate COVID collapse cannot be explained by a uniform loss of image information."
        )
        st.markdown(
            "- The headline accuracy is real on this test split.\n"
            "- Part of the COVID performance comes from non-anatomical cues.\n"
            "- External, source-separated validation is required before any clinical claim.\n"
            "- The model should remain an educational prototype."
        )

disclaimer()
footer("Trust and limitations")

