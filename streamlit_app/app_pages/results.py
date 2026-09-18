import altair as alt
import streamlit as st

from lib.metrics import class_metrics, comparison_frame, test_summary
from lib.paths import FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "05 · Results",
    "Fine-tuned EfficientNetB0 delivered the strongest presented result",
    "Current metrics are loaded from the saved evaluation artifacts, not copied from report text.",
)

best = test_summary("transfer_learning/transfer_efficientnetb0_finetuned_metrics.json")
narrative(
    "Find a model that performs consistently across all four classes on held-out data.",
    "Compared model families using the same test split and class-sensitive metrics.",
    f"Fine-tuned EfficientNetB0 reached {best['accuracy']:.1%} accuracy and {best['macro_f1']:.1%} macro F1.",
)

with st.container(horizontal=True):
    st.metric("Test accuracy", f"{best['accuracy']:.1%}", border=True)
    st.metric("Macro F1", f"{best['macro_f1']:.1%}", border=True)
    st.metric("COVID recall", f"{best['covid_recall']:.1%}", border=True)

comparison = comparison_frame().melt(
    id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score"
)
chart = (
    alt.Chart(comparison)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("Score:Q", title="Held-out score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Model:N", title=None, sort="-x"),
        color=alt.Color("Metric:N", scale=alt.Scale(range=["#4C51BF", "#00798C"])),
        yOffset="Metric:N",
        tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".1%")],
    )
    .properties(height=300)
)
st.altair_chart(chart)

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

footer("Results")
