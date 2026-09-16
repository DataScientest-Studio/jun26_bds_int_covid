import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.metrics import load_metrics_json, per_class_table, split_summary
from lib.paths import BASELINE_REPORTS_DIR
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, safe_image

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

configure_page("Classical & Baseline Models")

page_header(
    "Classical & Baseline Models",
    "Before touching deep learning: how far can a dummy classifier, logistic regression, "
    "and gradient boosting get on downsampled raw pixels alone?",
)

narrative_row(
    challenge=(
        "Before touching deep learning, what is the honest floor? What can trivial models "
        "already do on raw pixels alone?"
    ),
    did=(
        "Trained a majority-class dummy classifier, logistic regression, and histogram "
        "gradient boosting on 64\u00d764 downsampled grayscale pixels, then reran logistic "
        "regression and gradient boosting on lungs-only and background-only masked crops "
        "to test for a region confound."
    ),
    found=(
        "A linear model reached 71.4% accuracy, and gradient boosting on raw pixels hit "
        "86.0%, higher than the naive full-image CNN. Background-only pixels alone got "
        "gradient boosting to 84.3% accuracy, a humbling sign that much of the "
        "separability does not require the lungs at all."
    ),
)

dummy_metrics = load_metrics_json(BASELINE_REPORTS_DIR / "dummy_metrics.json")
logreg_metrics = load_metrics_json(BASELINE_REPORTS_DIR / "logistic_regression_metrics.json")
hgb_metrics = load_metrics_json(BASELINE_REPORTS_DIR / "hist_gradient_boosting_metrics.json")

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Dummy (majority class)", "48.3%", "always predicts Normal")
metric_cols[1].metric("Logistic Regression", "71.4%", "71.4% macro F1")
metric_cols[2].metric("Hist Gradient Boosting", "86.0%", "beats naive full-image CNN")
metric_cols[3].metric("Input resolution", "64\u00d764", "4,096 flattened pixels")

comparison_tab, per_class_tab, confound_tab, matrix_tab = st.tabs(
    ["Model Comparison", "Per-Class Performance", "Region-Confound Protocol", "Confusion Matrices"]
)

with comparison_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        rows = []
        for name, metrics in [
            ("Dummy (majority class)", dummy_metrics),
            ("Logistic Regression", logreg_metrics),
            ("Hist Gradient Boosting", hgb_metrics),
        ]:
            summary = split_summary(metrics, split="test")
            if summary is None:
                continue
            rows.append(
                {
                    "Test accuracy": summary["accuracy"],
                    "Macro F1": summary["macro_f1"],
                }
            )
        chart_frame = pd.DataFrame(
            rows,
            index=["Dummy", "Logistic Regression", "Hist Gradient Boosting"],
        )
        st.bar_chart(chart_frame, height=340)
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Model": ["Dummy", "Logistic Regression", "Hist Gradient Boosting"],
                    "Test accuracy": ["48.29%", "71.38%", "86.01%"],
                    "Macro F1": ["0.163", "0.714", "0.868"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=145,
        )
        st.caption(
            "Same stratified 70/15/15 split and seed (42) as the rest of the pipeline. No "
            "CLAHE, lung masking, or augmentation applied at this stage."
        )
        key_finding(
            "Gradient boosting on raw pixels (86.0%) already outperforms the naive "
            "full-image simple CNN (81.96%) trained later in the pipeline."
        )

with per_class_tab:
    model_choice = st.radio(
        "Model",
        ["Logistic Regression", "Hist Gradient Boosting"],
        horizontal=True,
    )
    chosen_metrics = logreg_metrics if model_choice == "Logistic Regression" else hgb_metrics
    left, right = st.columns([3, 2], gap="medium")
    with left:
        table = per_class_table(chosen_metrics, split="test", class_names=CLASS_NAMES)
        if table is not None:
            display_table = table.copy()
            for column in ["Precision", "Recall", "F1-score"]:
                display_table[column] = display_table[column].map(lambda v: f"{v:.3f}")
            st.dataframe(display_table, hide_index=True, width="stretch", height=180)
    with right:
        st.caption(
            "The dummy classifier always predicts Normal, so it scores 0.0 precision, "
            "recall, and F1 on every other class by construction, the trivial floor every "
            "later model must clear."
        )
        if model_choice == "Logistic Regression":
            st.caption(
                "Viral Pneumonia gets the best recall (90.0%) despite being the smallest "
                "class, likely thanks to class-weighted training. COVID is the weakest "
                "class for a linear model (63.4% recall)."
            )
        else:
            st.caption(
                "Gradient boosting is far more balanced across classes: COVID recall jumps "
                "to 84.9% and every class clears 78% F1, without any lung-specific "
                "features."
            )

with confound_tab:
    st.caption(
        "Same 64\u00d764 downsample-and-flatten pipeline, but the lung segmentation mask "
        "zeroes out either the background (lungs-only) or the lung field itself "
        "(background-only, the confound estimate)."
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(BASELINE_REPORTS_DIR / "mask_sanity_check.png", width="stretch")
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Model": ["Logistic Regression", "Hist Gradient Boosting"],
                    "Full image": ["0.714", "0.868"],
                    "Lungs only": ["0.562", "0.751"],
                    "Background only": ["0.660", "0.855"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=110,
        )
        key_finding(
            "Background-only gradient boosting (macro F1 0.855) nearly matches the full "
            "image (0.868) and clearly beats lungs-only (0.751). Non-pulmonary pixels "
            "carry most of the separability this model finds."
        )
        st.caption(
            "Macro F1 shown for the test split. The background column is the confound "
            "estimate: how well the class can be predicted from pixels that contain no "
            "lung tissue at all."
        )

with matrix_tab:
    cols = st.columns(3, gap="medium")
    for col, (name, filename) in zip(
        cols,
        [
            ("Dummy", "dummy_test_confusion_matrix.png"),
            ("Logistic Regression", "logistic_regression_test_confusion_matrix.png"),
            ("Hist Gradient Boosting", "hist_gradient_boosting_test_confusion_matrix.png"),
        ],
    ):
        with col:
            st.markdown(f"**{name}**")
            safe_image(BASELINE_REPORTS_DIR / filename, width="stretch")

app_footer()
