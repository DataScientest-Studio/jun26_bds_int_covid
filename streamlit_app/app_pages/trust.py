import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import disclaimer, figure, footer, narrative, page_intro


page_intro(
    "05 · Trust and limitations",
    "A strong score can still come from the wrong evidence",
    "Grad-CAM raised the concern; a sequence of constrained experiments tested it.",
)

narrative(
    "Determine whether predictions depend on lung pathology or on source-specific background cues.",
    "Inspected Grad-CAM, then tried lung crops, masks, lungs-only inputs, mask-only inputs, and masked pooling.",
    "Constraints reduced but did not fully remove the source-related shortcut.",
)

with st.container(horizontal=True):
    st.metric("Full-image COVID recall", "90.3%", border=True)
    st.metric("Lungs-only COVID recall", "57.0%", "−33.3 points", border=True)
    st.metric("Overall accuracy change", "−6.6 points", border=True)

st.markdown("#### Reasoning chain")
steps = st.columns(4, gap="small")
reasoning = [
    ("1", "Observe", "Grad-CAM often highlights borders, shoulders, and background."),
    ("2", "Intervene", "Try ROI crops, masks, lungs-only inputs, and masked pooling."),
    ("3", "Compare", "The source-related signal persists across constrained variants."),
    ("4", "Conclude carefully", "Evidence of shortcut learning—not proof of background-only prediction."),
]
for column, (number, title, body) in zip(steps, reasoning):
    with column.container(border=True, height="stretch"):
        st.caption(number)
        st.markdown(f"**{title}**")
        st.write(body)

tab1, tab2, tab3 = st.tabs(["Grad-CAM evidence", "What we tried", "Matched intervention"])
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
    st.caption(
        "Grad-CAM is coarse and correlational: it shows influential regions, not a lesion and not causal proof."
    )
with tab2:
    attempts = [
        ("Lung ROI crops", "Reduced visible background but retained framing and source-linked anatomy/geometry."),
        ("Pixel masking", "Removed non-lung pixels and caused a class-specific performance collapse."),
        ("Lungs-only inputs", "Kept lung texture; did not recover the full-image COVID recall."),
        ("Mask-only inputs", "Tested geometry alone; COVID recall was only 29.3%."),
        ("Masked pooling", "Constrained feature aggregation; raw performance improved, but source bias was not proven resolved."),
    ]
    for title, body in attempts:
        with st.container(border=True):
            st.markdown(f"**{title}** — {body}")
    st.info(
        "Answer to ‘What did you try?’: several anatomical constraints, not just one visualization or one mask.",
        icon=":material/check_circle:",
    )
with tab3:
    left, right = st.columns(2, gap="large")
    with left:
        figure(
            FIGURES_DIR / "gradcam_transfer_masked.png",
            "After masking the background, the model is forced to rely on lung content.",
        )
    with right:
        st.subheader("What this establishes")
        st.write(
            "In the matched EfficientNet experiment, the architecture, split, and seed stayed fixed while access to non-lung pixels changed. The disproportionate COVID collapse is evidence that the full-image model exploited COVID-specific information outside the lungs."
        )
        st.markdown(
            "- The headline accuracy is real on this test split.\n"
            "- Part of its COVID performance depends on non-lung cues.\n"
            "- The model still retains useful lung-derived signal.\n"
            "- External, source-separated validation is required before any clinical claim.\n"
            "- The model should remain an educational prototype."
        )
        st.warning(
            "Say ‘evidence of shortcut learning,’ not ‘proof that the model uses only the background.’",
            icon=":material/record_voice_over:",
        )

disclaimer()
footer("Trust and limitations")
