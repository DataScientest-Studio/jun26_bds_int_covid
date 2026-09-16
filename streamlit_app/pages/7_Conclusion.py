import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.metrics import build_comparison_table, load_metrics_json, split_summary
from lib.paths import BASELINE_REPORTS_DIR, CNN_REPORTS_DIR, TRANSFER_REPORTS_DIR
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, step_card

configure_page("Conclusion")

page_header(
    "Conclusion",
    "Scientific takeaways, honest limits, and what external validation would be "
    "required before any of this could be trusted.",
)

narrative_row(
    challenge=(
        "Can image classification distinguish COVID-19 chest X-rays from Normal, "
        "Viral Pneumonia, and Lung Opacity on a public Kaggle collection—and can "
        "we trust the headline numbers?"
    ),
    did=(
        "Ran the full pipeline from EDA through classical baselines, from-scratch "
        "and transfer CNNs, then Grad-CAM plus causal background-masking and "
        "shape-only ablations to test where accuracy actually comes from."
    ),
    found=(
        "Strong in-distribution accuracy is achievable (93.78% best test accuracy "
        "from scratch), but a large share of COVID detection on full images depends "
        "on non-lung shortcuts. High accuracy alone is not evidence of "
        "trustworthiness."
    ),
)

comparison_entries = [
    {"name": "Hist Gradient Boosting", "family": "Classical", "metrics_path": BASELINE_REPORTS_DIR / "hist_gradient_boosting_metrics.json"},
    {"name": "Simple CNN (full)", "family": "CNN scratch", "metrics_path": CNN_REPORTS_DIR / "cnn_simple_metrics.json"},
    {"name": "Scratch CNN (VGG-style)", "family": "CNN scratch", "metrics_path": CNN_REPORTS_DIR / "cnn_scratch_metrics.json"},
    {"name": "EfficientNetB0 frozen", "family": "Transfer", "metrics_path": TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_metrics.json"},
    {"name": "EfficientNetB0 fine-tuned", "family": "Transfer", "metrics_path": TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_finetuned_metrics.json"},
    {"name": "EfficientNetB0 lungs only", "family": "Causal ablation", "metrics_path": TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_masked_metrics.json"},
]

comparison_frame = build_comparison_table(comparison_entries, split="test")
best_row = None
if not comparison_frame.empty:
    best_row = comparison_frame.loc[comparison_frame["Test accuracy"].idxmax()]

scratch_metrics = load_metrics_json(CNN_REPORTS_DIR / "cnn_scratch_metrics.json")
scratch_summary = split_summary(scratch_metrics, split="test")
masked_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_masked_metrics.json")

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric(
    "Best test accuracy",
    f"{100 * scratch_summary['accuracy']:.2f}%" if scratch_summary else "—",
    "Scratch CNN (full image)",
)
metric_cols[1].metric("Models trained", "13+", "across 4 families")
metric_cols[2].metric(
    "COVID recall drop",
    "−33.3 pts",
    "full image → lungs only",
)
metric_cols[3].metric(
    "Classical ceiling",
    "86.0%",
    "gradient boosting on raw pixels",
)

results_tab, takeaways_tab, deployment_tab, next_steps_tab = st.tabs(
    ["Results Overview", "Scientific Takeaways", "Deployment Framing", "What We Would Do Differently"]
)

with results_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        if not comparison_frame.empty:
            display = comparison_frame.copy()
            display["Test accuracy"] = display["Test accuracy"].map(lambda value: f"{100 * value:.2f}%")
            display["Macro F1"] = display["Macro F1"].map(lambda value: f"{value:.3f}")
            st.dataframe(display, hide_index=True, width="stretch", height=220)
            chart = comparison_frame.set_index("Model")[["Test accuracy", "Macro F1"]]
            st.bar_chart(chart, height=320)
    with right:
        st.caption(
            "Numbers are read from saved metrics JSON under reports/, not retyped "
            "from the LaTeX report. The from-scratch CNN now outperforms the "
            "fine-tuned transfer model described elsewhere as the project best."
        )
        if best_row is not None:
            key_finding(
                f"Highest raw test accuracy: {best_row['Model']} "
                f"({100 * best_row['Test accuracy']:.2f}%, macro F1 "
                f"{best_row['Macro F1']:.3f}). Interpret alongside Step 6 "
                f"shortcut evidence before treating it as the best model."
            )

with takeaways_tab:
    cols = st.columns(2, gap="medium")
    with cols[0]:
        step_card(
            "1",
            "Separability exists early",
            "Gradient boosting on 64×64 raw pixels reached 86.0% accuracy—above "
            "the naive full-image CNN—and background-only pixels alone reached "
            "84.3% macro F1.",
            status="done",
        )
        step_card(
            "3",
            "Pretrained weights help, not guaranteed",
            "EfficientNetB0 fine-tuning reached 92.36% test accuracy but still "
            "trailed the from-scratch VGG-style CNN on this dataset.",
            status="done",
        )
    with cols[1]:
        step_card(
            "2",
            "Region matters",
            "Lung-centred ROI cropping cost 3.3 accuracy points and nearly 13 COVID "
            "recall points for the simple CNN, foreshadowing the causal masking result.",
            status="done",
        )
        step_card(
            "4",
            "Trust ≠ accuracy",
            "Background masking collapsed COVID recall from 90.3% to 57.0%. The "
            "headline number hides dataset-specific shortcuts concentrated in one class.",
            status="done",
        )

with deployment_tab:
    st.markdown(
        "This project is an **educational decision-support study**, not a clinically "
        "validated diagnostic system. Nothing here is medical advice.\n\n"
        "Before any deployment-oriented use, we would require at minimum:\n\n"
        "1. **External validation** on X-rays from hospitals and scanners not "
        "represented in the Kaggle collection\n"
        "2. **Prospective evaluation** with predefined thresholds and operating points\n"
        "3. **Subgroup reporting** by acquisition source, age, and severity where metadata exists\n"
        "4. **Uncertainty quantification** so low-confidence cases can be flagged\n"
        "5. **Monitoring for COVID-recall collapse** on new data—the predicted failure mode "
        "when background shortcuts no longer hold\n\n"
        "The lungs-only masked model (83.07% accuracy, 57.0% COVID recall) is a more "
        "honest lower bound on what lung tissue alone supports with this architecture, "
        "at the cost of lower raw performance."
    )
    baseline_metrics = load_metrics_json(TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_metrics.json")
    if baseline_metrics and masked_metrics:
        baseline_report = baseline_metrics.get("test", {}).get("report", {})
        masked_report = masked_metrics.get("test", {}).get("report", {})
        recall_rows = []
        for class_name in ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]:
            recall_rows.append(
                {
                    "Class": class_name,
                    "Full-image recall": f"{baseline_report[class_name]['recall']:.3f}",
                    "Lungs-only recall": f"{masked_report[class_name]['recall']:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(recall_rows), hide_index=True, width="stretch", height=180)

with next_steps_tab:
    st.markdown(
        "**Data and protocol**\n\n"
        "- Source-aware splits and duplicate review beyond exact pixel matches\n"
        "- Drop horizontal flip as a default augmentation on chest X-rays\n"
        "- Report full-image and lungs-only metrics together in every model card\n\n"
        "**Modeling**\n\n"
        "- Explore architectures and losses that penalize background attention explicitly\n"
        "- Compare against the shape-only baseline before claiming pulmonary signal\n"
        "- Fine-tune with domain-specific pretraining if larger medical corpora become available\n\n"
        "**Product / presentation**\n\n"
        "- Keep this Streamlit walkthrough as the self-paced defense artifact\n"
        "- Use the Live Demo page for hands-on closing, with the non-clinical disclaimer visible"
    )

app_footer()
