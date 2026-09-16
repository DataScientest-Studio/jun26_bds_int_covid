import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.metrics import load_metrics_json, per_class_table, split_summary
from lib.paths import TRANSFER_REPORTS_DIR
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, safe_image

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

configure_page("Transfer Learning")

page_header(
    "Transfer Learning",
    "EfficientNetB0 pretrained on ImageNet, adapted to grayscale chest X-rays through a "
    "frozen backbone, augmentation and class-weighting ablations, then end-to-end "
    "fine-tuning.",
)

narrative_row(
    challenge=(
        "Does ImageNet-pretrained knowledge transfer to grayscale medical X-rays, a very "
        "different visual domain from natural photographs?"
    ),
    did=(
        "Trained an EfficientNetB0 frozen backbone with a new classification head, ran "
        "augmentation (with and without horizontal flip) and class-weighting ablations on "
        "top of it, then unfroze the backbone for end-to-end fine-tuning."
    ),
    found=(
        "The frozen backbone alone hit 89.7% accuracy; horizontal flip hurt performance, "
        "since chest X-rays are not meaningfully flip-invariant; fine-tuning reached "
        "92.36%, a strong result but still short of the from-scratch CNN in Step 4."
    ),
)

frozen_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_metrics.json")
augmented_metrics = load_metrics_json(
    TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_metrics.json"
)
no_flip_metrics = load_metrics_json(
    TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_no_flip_metrics.json"
)
class_weighted_metrics = load_metrics_json(
    TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_class_weighted_metrics.json"
)
finetuned_metrics = load_metrics_json(
    TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_finetuned_metrics.json"
)

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Frozen Backbone", "89.70%", "0.907 macro F1")
metric_cols[1].metric("Fine-Tuned", "92.36%", "+2.66 pts over frozen")
metric_cols[2].metric("Horizontal Flip", "-0.76 pts", "hurt test accuracy")
metric_cols[3].metric("Class-Weighted COVID Recall", "93.1%", "vs 90.3% frozen")

comparison_tab, per_class_tab, ablation_tab, finetune_tab, matrix_tab = st.tabs(
    [
        "Model Comparison",
        "Per-Class Performance",
        "Augmentation & Class Weighting",
        "Fine-Tuning",
        "Confusion Matrices",
    ]
)

MODEL_ENTRIES = [
    ("Frozen Backbone", frozen_metrics),
    ("Augmented (flip)", augmented_metrics),
    ("Augmented (no flip)", no_flip_metrics),
    ("Class-Weighted", class_weighted_metrics),
    ("Fine-Tuned", finetuned_metrics),
]

with comparison_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        rows = []
        labels = []
        for label, metrics in MODEL_ENTRIES:
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
                    "Model": [label for label, _ in MODEL_ENTRIES],
                    "Test accuracy": ["89.70%", "87.62%", "88.38%", "88.34%", "92.36%"],
                    "Macro F1": ["0.907", "0.884", "0.896", "0.894", "0.936"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=215,
        )
        st.caption(
            "Same stratified 70/15/15 split and seed (42) as the rest of the pipeline. "
            "Every model shares the same architecture: GlobalAveragePooling, Dropout, "
            "Dense(128), Dropout, Dense(4, softmax) on top of EfficientNetB0."
        )
        key_finding(
            "Fine-tuning is the clear winner among these five runs, gaining 2.66 accuracy "
            "points and 2.9 macro-F1 points over the frozen baseline, more than any "
            "ablation on the frozen backbone achieved on its own."
        )

with per_class_tab:
    model_choice = st.radio(
        "Model",
        [label for label, _ in MODEL_ENTRIES],
        horizontal=True,
    )
    chosen_metrics = dict(MODEL_ENTRIES)[model_choice]
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
            "Lung_Opacity is the largest class and the hardest to separate for every "
            "variant; Viral Pneumonia, the smallest class, is consistently the easiest "
            "thanks to class-weighted training across the board."
        )
        key_finding(
            "Fine-tuning lifts COVID recall to 97.8%, the strongest of any model on this "
            "page, while Lung_Opacity recall barely moves (81.4% vs 80.8% frozen), showing "
            "the gain is concentrated in one class rather than spread evenly."
        )

with ablation_tab:
    st.caption(
        "Two independent ablations on the frozen backbone: does light augmentation help "
        "generalization, and does horizontal flip specifically help or hurt on chest "
        "X-rays that are not left/right symmetric in a clinically meaningless way?"
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_no_flip_history.png",
            width="stretch",
        )
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Model": ["Frozen Backbone", "Augmented (flip)", "Augmented (no flip)", "Class-Weighted"],
                    "Test accuracy": ["89.70%", "87.62%", "88.38%", "88.34%"],
                    "Macro F1": ["0.907", "0.884", "0.896", "0.894"],
                    "Train-test acc gap": ["2.77 pts", "2.72 pts", "1.98 pts", "3.30 pts"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        key_finding(
            "Removing horizontal flip improved every test metric over flip-augmented "
            "training and nearly halved the train-test generalization gap (1.98 vs 2.77 "
            "points), while class weighting raises COVID recall without shrinking that gap "
            "at all, it solves imbalance, not overfitting."
        )
        st.caption(
            "Train-test accuracy gap is train accuracy minus test accuracy: a rough "
            "overfitting signal, not a performance ranking on its own."
        )

with finetune_tab:
    st.caption(
        "Fine-tuning unfreezes the entire EfficientNetB0 backbone after the frozen-head "
        "phase and continues training end-to-end at a much lower learning rate "
        "(1e-5 vs 1e-3), letting the pretrained ImageNet filters adapt to X-ray textures."
    )
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_finetuned_history.png",
            width="stretch",
        )
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Test accuracy", "Test macro F1", "COVID recall", "Lung_Opacity recall"],
                    "Frozen Backbone": ["89.70%", "0.907", "90.3%", "80.8%"],
                    "Fine-Tuned": ["92.36%", "0.936", "97.8%", "81.4%"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        key_finding(
            "Fine-tuning is the strongest transfer-learning result on this page (92.36% "
            "accuracy), but it still falls short of the from-scratch CNN in Step 4, "
            "pretrained ImageNet features help, they do not guarantee the best outcome on "
            "this dataset."
        )

with matrix_tab:
    cols = st.columns(4, gap="medium")
    for col, (name, filename) in zip(
        cols,
        [
            ("Frozen Backbone", "transfer_efficientnetb0_test_confusion_matrix.png"),
            ("Augmented (no flip)", "transfer_efficientnetb0_augmented_no_flip_test_confusion_matrix.png"),
            ("Class-Weighted", "transfer_efficientnetb0_class_weighted_test_confusion_matrix.png"),
            ("Fine-Tuned", "transfer_efficientnetb0_finetuned_test_confusion_matrix.png"),
        ],
    ):
        with col:
            st.markdown(f"**{name}**")
            safe_image(TRANSFER_REPORTS_DIR / filename, width="stretch")

app_footer()
