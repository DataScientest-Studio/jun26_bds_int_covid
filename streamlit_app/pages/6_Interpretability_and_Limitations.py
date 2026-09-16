import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.metrics import load_metrics_json, per_class_table, split_summary
from lib.paths import SUMMARY_FIGURES_DIR, TRANSFER_REPORTS_DIR
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, safe_image

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

configure_page("Interpretability & Limitations")

page_header(
    "Interpretability & Limitations",
    "High accuracy is not evidence of trustworthiness. Is the model reading lungs "
    "or reading the dataset?",
)

narrative_row(
    challenge=(
        "Headline accuracy does not show whether the model learns pulmonary "
        "pathology or dataset-specific shortcuts from borders, positioning, and "
        "source-hospital signatures."
    ),
    did=(
        "Ran Grad-CAM with a lung-focus-vs-chance metric on EfficientNetB0, then "
        "a causal test that retrains with the background forcibly zeroed out, plus "
        "a shape-only ablation that feeds the lung mask silhouette alone."
    ),
    found=(
        "Masking the background collapsed COVID recall from 90.3% to 57.0% while "
        "other classes barely moved. Causal proof of dataset-specific shortcut "
        "learning, not just a correlational Grad-CAM hunch."
    ),
)

baseline_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_metrics.json")
masked_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_masked_metrics.json")
mask_only_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_mask_only_metrics.json")

baseline_summary = split_summary(baseline_metrics, split="test")
masked_summary = split_summary(masked_metrics, split="test")
mask_only_summary = split_summary(mask_only_metrics, split="test")

def recall_for(metrics, class_name: str) -> float:
    if not metrics:
        return float("nan")
    try:
        return metrics["test"]["report"][class_name]["recall"]
    except (KeyError, TypeError):
        return float("nan")

baseline_covid_recall = recall_for(baseline_metrics, "COVID")
masked_covid_recall = recall_for(masked_metrics, "COVID")
mask_only_covid_recall = recall_for(mask_only_metrics, "COVID")

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric(
    "Full-image test accuracy",
    f"{100 * baseline_summary['accuracy']:.2f}%" if baseline_summary else "—",
)
metric_cols[1].metric(
    "COVID recall (full image)",
    f"{100 * baseline_covid_recall:.1f}%",
    f"{100 * (masked_covid_recall - baseline_covid_recall):+.1f} pts when masked",
)
metric_cols[2].metric(
    "Lungs-only test accuracy",
    f"{100 * masked_summary['accuracy']:.2f}%" if masked_summary else "—",
    "background zeroed",
)
metric_cols[3].metric(
    "Shape-only COVID recall",
    f"{100 * mask_only_covid_recall:.1f}%",
    "mask silhouette input",
)

gradcam_tab, lung_focus_tab, causal_tab, decomposition_tab, limitations_tab = st.tabs(
    [
        "Grad-CAM",
        "Lung Focus vs Chance",
        "Causal Background Masking",
        "Accuracy Decomposition",
        "Limitations",
    ]
)

with gradcam_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_gradcam.png",
            width="stretch",
        )
    with right:
        st.caption(
            "Grad-CAM on the augmented EfficientNetB0 model: heatmaps often sit at "
            "chest borders, shoulders, and corners rather than tightly inside the "
            "green lung outline."
        )
        key_finding(
            "Qualitative Grad-CAM alone is suggestive. The causal masking "
            "experiment below is the definitive test because it does not depend "
            "on heatmap spatial resolution."
        )
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_masked_gradcam.png",
            width="stretch",
        )

with lung_focus_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_lung_focus.png",
            width="stretch",
        )
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"],
                    "Grad-CAM in lungs": ["30.8%", "28.5%", "29.8%", "17.7%"],
                    "Chance (lung area)": ["24.7%", "25.0%", "25.7%", "20.6%"],
                    "Gap": ["+6.1 pts", "+3.5 pts", "+4.2 pts", "−2.9 pts"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        st.caption(
            "Only mildly above chance for three classes; Lung_Opacity is below "
            "chance, meaning the model attends to background more than pure "
            "geometry would predict."
        )
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_augmented_lung_focus_by_correctness.png",
            width="stretch",
        )
        st.caption(
            "Correct and misclassified predictions show virtually identical "
            "lung-focus. Weak localization is baked into general behavior, not "
            "a special failure mode."
        )

with causal_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        table = per_class_table(baseline_metrics, split="test", class_names=CLASS_NAMES)
        masked_table = per_class_table(masked_metrics, split="test", class_names=CLASS_NAMES)
        if table is not None and masked_table is not None:
            comparison = table[["Class", "Recall"]].merge(
                masked_table[["Class", "Recall"]],
                on="Class",
                suffixes=(" (full)", " (lungs only)"),
            )
            display = comparison.copy()
            for column in display.columns[1:]:
                display[column] = display[column].map(lambda value: f"{value:.3f}")
            st.dataframe(display, hide_index=True, width="stretch", height=180)
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Test accuracy", "Macro F1", "COVID recall", "COVID F1"],
                    "Full image": ["89.70%", "0.907", "90.3%", "0.923"],
                    "Lungs only": ["83.07%", "0.823", "57.0%", "0.678"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        key_finding(
            "Removing non-lung pixels costs 6.6 accuracy points overall but 33.3 "
            "COVID recall points. Other classes lose only a few points each. "
            "COVID-specific background leakage, not uniform pathology signal."
        )
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_masked_test_confusion_matrix.png",
            width="stretch",
        )

with decomposition_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(TRANSFER_REPORTS_DIR / "all_models_test_comparison.png", width="stretch")
    with right:
        rows = []
        for label, metrics in [
            ("Full image", baseline_metrics),
            ("Lungs only (texture kept)", masked_metrics),
            ("Shape only (mask silhouette)", mask_only_metrics),
        ]:
            summary = split_summary(metrics, split="test")
            if summary is None:
                continue
            rows.append(
                {
                    "Input": label,
                    "Test accuracy": summary["accuracy"],
                    "Macro F1": summary["macro_f1"],
                    "COVID recall": recall_for(metrics, "COVID"),
                }
            )
        if rows:
            chart_frame = pd.DataFrame(rows).set_index("Input")
            st.bar_chart(chart_frame[["Test accuracy", "Macro F1"]], height=280)
        st.dataframe(
            pd.DataFrame(
                {
                    "Decomposition step": [
                        "Background / context",
                        "Lung texture (beyond shape)",
                        "Lung geometry alone",
                    ],
                    "Accuracy delta": ["−6.6 pts", "−9.0 pts", "74.1% floor"],
                    "COVID recall note": [
                        "90.3% → 57.0%",
                        "Texture needed beyond silhouette",
                        "29.3% (near random for 4 classes)",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
            height=150,
        )
        key_finding(
            "Shape-only input is the most anatomically conservative baseline and "
            "the worst on COVID. COVID classification in this dataset leans on "
            "cues outside lung outline as well as in-lung texture."
        )
        safe_image(
            TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_mask_only_test_confusion_matrix.png",
            width="stretch",
        )

with limitations_tab:
    left, right = st.columns([2, 3], gap="medium")
    with left:
        safe_image(SUMMARY_FIGURES_DIR / "region_confound_comparison.png", width="stretch")
    with right:
        st.markdown(
            "**Scientific limits**\n\n"
            "- Single public Kaggle collection stitched from heterogeneous sources per class\n"
            "- No external hospital or scanner validation\n"
            "- Grad-CAM is correlational; background masking is the causal evidence\n"
            "- Class imbalance persists (Viral Pneumonia ≈ 6% of training data)\n\n"
            "**Deployment framing**\n\n"
            "Before any clinical use, this line of models would need multi-site "
            "external validation, subgroup performance reporting, uncertainty "
            "quantification, and monitoring for COVID-recall collapse on new "
            "sources—the failure mode this analysis predicts.\n\n"
            "**What we would do differently**\n\n"
            "- Provenance-aware splitting by source hospital where metadata allows\n"
            "- Background randomization or explicit confound controls at collection time\n"
            "- Report lungs-only and full-image metrics side by side by default\n"
            "- Treat shortcut findings as central results, not appendix caveats"
        )

app_footer()
