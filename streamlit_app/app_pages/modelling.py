import pandas as pd
import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "03 · Modelling",
    "Eight experiments, each with one job",
    "The sequence measures performance and probes where that performance comes from.",
)

narrative(
    "Establish useful baselines, improve raw performance, then challenge the model's dependence on non-lung cues.",
    "Kept one split and evaluation protocol while changing model capacity or access to image regions.",
    "Fine-tuning won on raw score; region experiments exposed a separate trust problem.",
)

st.markdown("#### Compact experiment map")
experiments = pd.DataFrame(
    [
        ["1", "Dummy baseline", "Set the majority-class floor", "48.3% accuracy; macro F1 16.3%"],
        ["2", "Logistic regression", "Test linear signal in downsampled pixels", "71.4% accuracy; useful non-deep baseline"],
        ["3", "Simple CNN", "Test learned spatial features", "82.0% accuracy; clear gain over linear features"],
        ["4", "Base EfficientNetB0", "Measure frozen transfer features", "87.1% accuracy; transfer learning helped"],
        ["5", "Fine-tuned EfficientNetB0", "Adapt pretrained features", "92.4% accuracy; raw-score winner"],
        ["6", "Lung ROI EfficientNetB0", "Crop attention toward the chest", "84.3% accuracy; cropping did not remove the issue"],
        ["7", "Lungs-only EfficientNetB0", "Remove non-lung pixels", "COVID recall fell from 90.3% to 57.0% in the matched test"],
        ["8", "Confound probes", "Test background, mask shape, and masked pooling", "Residual source signal remained despite constraints"],
    ],
    columns=["#", "Experiment", "Purpose", "Main finding"],
)
st.dataframe(
    experiments,
    hide_index=True,
    column_config={"#": st.column_config.TextColumn(width="small")},
)
st.caption(
    "The defense names most rows briefly; it explains fine-tuning and the matched lungs-only test in detail."
)

tab1, tab2 = st.tabs(["Shared protocol", "Training evidence"])
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
                "- ROI, lungs-only, background, mask-only, and masked-pooling probes"
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
