import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "04 · Modelling",
    "We increased complexity only when the evidence justified it",
    "Each model family answers a different question, from a trivial floor to transferable visual features.",
)

narrative(
    "Establish whether deep learning adds value beyond class frequency and raw-pixel patterns.",
    "Compared dummy and linear baselines, gradient boosting, a compact CNN benchmark, and EfficientNetB0 transfer learning.",
    "Performance rose with learned spatial features, but model quality could not be judged from accuracy alone.",
)

st.markdown("#### Model progression")
columns = st.columns(4, gap="small")
cards = [
    ("01", "Dummy + logistic", "A transparent floor and a raw-pixel linear baseline."),
    ("02", "Boosted trees", "A nonlinear classical benchmark on compressed image features."),
    ("03", "Simple CNN", "A compact benchmark for learned spatial features."),
    ("04", "EfficientNetB0", "ImageNet transfer learning, then low-rate fine-tuning."),
]
for column, (number, title, body) in zip(columns, cards):
    with column.container(border=True, height="stretch"):
        st.caption(number)
        st.subheader(title)
        st.write(body)

st.space("small")

tab1, tab2 = st.tabs(["Experimental design", "Training evidence"])
with tab1:
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Evaluation protocol")
            st.markdown(
                "- Same deduplicated stratified split\n"
                "- 70% train / 15% validation / 15% test\n"
                "- Accuracy + macro F1 + per-class recall\n"
                "- Confusion matrices on the untouched test set"
            )
    with right:
        with st.container(border=True):
            st.subheader("Optimization decisions")
            st.markdown(
                "- Class weights and balanced sampling\n"
                "- Rotation-only augmentation\n"
                "- Frozen and fine-tuned backbones\n"
                "- Region ablations and Grad-CAM"
            )
with tab2:
    figure(
        FIGURES_DIR / "transfer_finetuned_history.png",
        "Fine-tuned EfficientNetB0 learning curves: the mid-run reset marks the transition into fine-tuning.",
    )
    st.caption(
        "Loss, accuracy, and macro F1 are shown together so model selection is not driven by accuracy alone."
    )

footer("Modelling")
