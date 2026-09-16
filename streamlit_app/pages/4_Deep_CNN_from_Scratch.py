import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.metrics import load_metrics_json, per_class_table, split_summary
from lib.paths import CNN_REPORTS_DIR
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, safe_image

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

configure_page("Deep CNN From Scratch")

page_header(
    "Deep CNN From Scratch",
    "Two low-capacity architectures, LeNet-5 and a lung-centred ROI CNN, trained entirely "
    "from random weights, no pretrained backbone anywhere on this page.",
)

narrative_row(
    challenge=(
        "With no pretrained weights available, how much do architecture capacity and "
        "image region actually matter for a CNN trained from scratch?"
    ),
    did=(
        "Trained LeNet-5 (the classic low-capacity floor, full image) and a lung-centred "
        "ROI variant of the same simple 3-block CNN used as the full-image baseline, all "
        "under the same split, seed, and class-weighting strategy."
    ),
    found=(
        "LeNet-5 lands within a point of the full-image simple CNN despite a 65-year-old, "
        "position-sensitive design; cropping that same simple CNN to a lung-centred ROI "
        "costs 3.3 accuracy points and drops COVID recall by nearly 13 points, an early "
        "hint of a background shortcut confirmed causally in Step 6."
    ),
)

simple_metrics = load_metrics_json(CNN_REPORTS_DIR / "cnn_simple_metrics.json")
lenet_metrics = load_metrics_json(CNN_REPORTS_DIR / "lenet_metrics.json")
roi_metrics = load_metrics_json(CNN_REPORTS_DIR / "cnn_simple_roi_baseline_lung_roi_metrics.json")

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Simple CNN (full image)", "81.96%", "0.827 macro F1")
metric_cols[1].metric("LeNet-5", "81.71%", "61K parameters")
metric_cols[2].metric("Lung-ROI CNN", "78.71%", "23,556 parameters, cropped")
metric_cols[3].metric("COVID recall drop", "-12.7 pts", "full image \u2192 lung ROI")

comparison_tab, per_class_tab, region_tab, matrix_tab = st.tabs(
    ["Architecture Comparison", "Per-Class Performance", "Region Ablation", "Confusion Matrices"]
)

with comparison_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        rows = []
        labels = []
        for label, metrics in [
            ("Simple CNN", simple_metrics),
            ("LeNet-5", lenet_metrics),
            ("Lung-ROI CNN", roi_metrics),
        ]:
            summary = split_summary(metrics, split="test")
            if summary is None:
                continue
            labels.append(label)
            rows.append(
                {
                    "Test accuracy": summary["accuracy"],
                    "Macro F1": summary["macro_f1"],
                }
            )
        chart_frame = pd.DataFrame(rows, index=labels)
        st.bar_chart(chart_frame, height=340)
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Model": ["Simple CNN", "LeNet-5", "Lung-ROI CNN"],
                    "Parameters": ["23,556", "61,196", "23,556"],
                    "Test accuracy": ["81.96%", "81.71%", "78.71%"],
                    "Macro F1": ["0.827", "0.815", "0.774"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=150,
        )
        st.caption(
            "Same stratified 70/15/15 split and seed (42) as the rest of the pipeline. "
            "All three architectures are trained from random initialization, no "
            "pretrained weights anywhere on this page."
        )
        key_finding(
            "Simple CNN and Lung-ROI CNN share the exact same architecture and parameter "
            "count (23,556); the only difference is whether the input includes peripheral "
            "background. That alone costs 3.3 accuracy points and 5.3 macro-F1 points, a "
            "controlled hint that peripheral content matters."
        )

with per_class_tab:
    model_choice = st.radio(
        "Model",
        ["LeNet-5", "Lung-ROI CNN"],
        horizontal=True,
    )
    chosen_metrics = lenet_metrics if model_choice == "LeNet-5" else roi_metrics
    left, right = st.columns([3, 2], gap="medium")
    with left:
        table = per_class_table(chosen_metrics, split="test", class_names=CLASS_NAMES)
        if table is not None:
            display_table = table.copy()
            for column in ["Precision", "Recall", "F1-score"]:
                display_table[column] = display_table[column].map(lambda v: f"{v:.3f}")
            st.dataframe(display_table, hide_index=True, width="stretch", height=180)
    with right:
        if model_choice == "LeNet-5":
            st.caption(
                "LeNet-5's fully-connected head can key on absolute pixel position, not "
                "just texture, and it still lands within a point of the full-image simple "
                "CNN. COVID is its weakest class at 73.0% F1."
            )
        else:
            st.caption(
                "Restricting to the lung-centred crop hurts COVID and Lung Opacity the "
                "most (67.5% and 75.1% F1) while Normal and Viral Pneumonia barely move, "
                "foreshadowing the causal background-masking result in Step 6."
            )
        key_finding(
            "COVID is the weakest class for both architectures, and it gets weaker still "
            "once the lung-ROI crop removes peripheral context (73.0% F1 full image vs "
            "67.5% F1 cropped), the opposite of what a purely pulmonary signal would "
            "predict."
        )

with region_tab:
    st.caption(
        "The lung-ROI CNN reuses the simple architecture (23,556 parameters) but crops "
        "each image to a lung-centred square before resizing, using the segmentation mask "
        "only to find the crop, never to zero out pixels."
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(CNN_REPORTS_DIR / "lung_roi_preprocessing_check.png", width="stretch")
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Region": ["Full image", "Lung-centred ROI"],
                    "Test accuracy": ["81.96%", "78.71%"],
                    "Macro F1": ["0.827", "0.774"],
                    "COVID recall": ["83.0%", "70.3%"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=110,
        )
        key_finding(
            "Cropping out peripheral background costs 3.3 accuracy points and drops COVID "
            "recall by nearly 13 points, an early sign that some of the simple CNN's "
            "separability lives outside the lungs, the same shortcut the causal test in "
            "Step 6 confirms directly."
        )
        st.caption(
            "An earlier hard lung-masking variant (zeroing the background outright) fell "
            "further, to 72.93% accuracy, but introduced sharp artificial edges at the "
            "mask boundary. The soft ROI crop above avoids that artefact while still "
            "reducing peripheral content."
        )

with matrix_tab:
    cols = st.columns(3, gap="medium")
    for col, (name, filename) in zip(
        cols,
        [
            ("Simple CNN", "cnn_simple_test_confusion_matrix.png"),
            ("LeNet-5", "lenet_test_confusion_matrix.png"),
            ("Lung-ROI CNN", "cnn_simple_roi_baseline_lung_roi_test_confusion_matrix.png"),
        ],
    ):
        with col:
            st.markdown(f"**{name}**")
            safe_image(CNN_REPORTS_DIR / filename, width="stretch")

app_footer()
