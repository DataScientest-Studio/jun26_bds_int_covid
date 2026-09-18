import streamlit as st

from lib.metrics import class_metrics, test_summary
from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "04 · Results",
    "Raw-score winner: fine-tuned EfficientNetB0",
    "The winning score is real on this split; whether it is trustworthy is a separate question.",
)

best = test_summary("transfer_learning/transfer_efficientnetb0_finetuned_metrics.json")
narrative(
    "Identify the strongest retained model on the untouched internal test split.",
    "Ranked representative models by accuracy, then checked macro F1 and class-level errors.",
    f"Fine-tuned EfficientNetB0 won with {best['accuracy']:.1%} accuracy and {best['macro_f1']:.1%} macro F1.",
)

with st.container(horizontal=True):
    st.metric("Test accuracy", f"{best['accuracy']:.1%}", border=True)
    st.metric("Macro F1", f"{best['macro_f1']:.1%}", border=True)
    st.metric("COVID recall", f"{best['covid_recall']:.1%}", border=True)

left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
with left:
    figure(
        FIGURES_DIR / "transfer_finetuned_confusion.png",
        "Fine-tuned EfficientNetB0 test confusion matrix; Lung Opacity vs Normal remains the main ambiguity.",
    )
with right:
    st.markdown("#### Fine-tuned EfficientNetB0 by class")
    st.dataframe(
        class_metrics("transfer_learning/transfer_efficientnetb0_finetuned_metrics.json"),
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "F1": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption("All values come from the saved fine-tuned EfficientNetB0 evaluation artifact.")

with st.container(border=True):
    st.subheader("Raw score ≠ trustworthiness")
    st.write(
        "This page answers **which retained model scored highest on the internal split**. It does not establish that the model learned clinically meaningful lung evidence or will generalize across hospitals and acquisition sources."
    )
    st.info(
        "Next: use Grad-CAM and controlled region experiments to investigate shortcut learning.",
        icon=":material/arrow_forward:",
    )

footer("Results")
