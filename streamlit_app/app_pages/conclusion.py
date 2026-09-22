import streamlit as st

from lib.ui import disclaimer, footer, page_intro


page_intro(
    "06 · Conclusion",
    "The score is strong; the claim must stay narrow",
    "The project succeeds as a reproducible classifier and as an audit of why internal performance can mislead.",
)

st.markdown("#### Answers to our two questions")
columns = st.columns(3, gap="medium")
sections = [
    (
        ":material/analytics:",
        "Performance",
        "Transfer learning clearly helps: 0.827 macro F1 from a small CNN trained from scratch, "
        "0.907 with a frozen EfficientNetB0, and 0.936 after fine-tuning—on par with published "
        "results on this dataset.",
    ),
    (
        ":material/warning:",
        "Attribution",
        "A large share of that performance does not come from the lungs. Every model does better "
        "on the background than on the lungs, and a 16 × 16 thumbnail with the lungs removed "
        "still reaches 0.787.",
    ),
    (
        ":material/next_plan:",
        "Next step",
        "Test on images from new hospitals first. Only once performance survives there is it worth "
        "optimising architectures further.",
    ),
]
for column, (icon, title, body) in zip(columns, sections):
    with column.container(border=True, height="stretch"):
        st.markdown(f"#### {icon} {title}")
        st.write(body)

st.markdown("#### Future work")
future = [
    (
        "External validation",
        "Evaluate on hospitals absent from this dataset. Our results make a testable prediction: "
        "COVID recall should drop far more than the other classes.",
    ),
    (
        "Predict the source directly",
        "Train a classifier whose label is the repository, not the diagnosis. Its accuracy measures "
        "how visible the source is in the pixels.",
    ),
    (
        "Source-adversarial training",
        "Penalise features that reveal the source, then repeat the background-versus-lungs comparison.",
    ),
    (
        "Test lung-constrained architectures properly",
        "Masked pooling kept most of the performance, but deep features still summarise the whole "
        "image. Perturbing the background would show whether it removes the shortcut or hides it.",
    ),
    (
        "Build datasets differently",
        "Draw every class from several hospitals, and every hospital from several classes—the most "
        "durable fix is not a modelling one.",
    ),
]
for title, detail in future:
    with st.container(border=True):
        st.markdown(f"**{title}**  ")
        st.caption(detail)

st.success(
    "Final takeaway: high accuracy on this dataset does not by itself show that a model recognises "
    "disease. Where the performance comes from has to be part of the evaluation.",
    icon=":material/flag:",
)
disclaimer()
footer("Conclusion")