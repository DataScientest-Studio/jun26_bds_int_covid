import pandas as pd
import streamlit as st

from lib.paths import FIGURES_DIR
from lib.ui import disclaimer, figure, footer, narrative, page_intro


page_intro(
    "05 · Trust and limitations",
    "A strong score can still come from the wrong evidence",
    "Grad-CAM raised the concern; a sequence of constrained experiments tested it.",
)

narrative(
    "Determine whether predictions depend on lung pathology or on source-specific cues.",
    "Inspected Grad-CAM, constrained the input in several ways, then ablated region and resolution across architectures.",
    "Every model does better on the background than on the lungs, and performance survives 16 × 16 inputs.",
)

with st.container(horizontal=True):
    st.metric("Full-image COVID recall", "90.3%", border=True)
    st.metric("Lungs-only COVID recall", "57.0%", "-33.3 points", border=True)
    st.metric("Background-only macro F1", "0.876", "-0.031 vs full image", border=True)

gradcam, attempts_tab, region, resolution, limits = st.tabs(
    ["Grad-CAM evidence", "What we tried", "Background vs lungs", "Resolution", "Limitations"]
)

with gradcam:
    left, right = st.columns(2, gap="large")
    with left:
        figure(
            FIGURES_DIR / "gradcam_transfer_baseline.png",
            "Full-image model: activation frequently extends to borders and shoulders.",
        )
    with right:
        figure(
            FIGURES_DIR / "transfer_lung_focus_baseline.png",
            "Attention inside the lungs is only mildly above chance and below chance for Lung Opacity.",
        )
    st.caption(
        "Grad-CAM is coarse and correlational: it shows influential regions, not a lesion and not causal proof. "
        "For the fine-tuned model, attention inside the lungs on COVID images was 24.3% against a chance level of 24.4%."
    )

with attempts_tab:
    attempts = [
        ("Lung ROI crops", "Reduced visible background but retained framing and source-linked anatomy/geometry."),
        ("Pixel masking", "Removed non-lung pixels and caused a class-specific performance collapse."),
        ("Lungs-only inputs", "Kept lung texture; did not recover the full-image COVID recall."),
        ("Mask-only inputs", "Tested geometry alone; COVID recall was only 29.3%."),
        (
            "Masked pooling",
            "Recovered most of the performance, but deep features over the lungs still summarise "
            "much of the image—so this may hide the shortcut rather than remove it.",
        ),
    ]
    for title, body in attempts:
        with st.container(border=True):
            st.markdown(f"**{title}** — {body}")


with region:
    st.markdown("#### Same model, different pixels")
    st.write(
        "Each model was trained three times: on the whole image, on the lungs only, and on "
        "everything **except** the lungs. Excluded pixels are set to zero, so all three inputs "
        "have the same size."
    )
    region_frame = pd.DataFrame(
        [
            ["EfficientNetB0 · 224²", 0.907, 0.823, 0.876],
            ["Simple CNN · 128²", 0.827, 0.690, 0.741],
            ["Simple CNN · 16²", 0.835, 0.749, 0.787],
            ["LeNet-5 · 32²", 0.815, 0.739, 0.747],
        ],
        columns=["Model", "Full", "Lungs", "Background"],
    )
    region_frame["Bg − Lungs"] = (region_frame["Background"] - region_frame["Lungs"]).round(3)
    score_column = st.column_config.ProgressColumn(format="%.3f", min_value=0.6, max_value=1.0)
    st.dataframe(
        region_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Full": score_column,
            "Lungs": score_column,
            "Background": score_column,
            "Bg − Lungs": st.column_config.NumberColumn(format="+%.3f"),
        },
    )
    st.caption("Test macro F1. EfficientNetB0 uses a frozen backbone. The final column is positive in every row.")
    st.warning(
        "In every architecture, the pixels **outside** the lungs are more informative than the lungs "
        "themselves. The pretrained EfficientNet shows the largest gap—more capacity does not protect "
        "against the shortcut.",
        icon=":material/warning:",
    )
    st.caption(
        "This matches the data audit: all COVID images come from COVID-only repositories, so recognising "
        "the source is enough to recognise COVID."
    )

with resolution:
    st.markdown("#### Detail is not needed")
    left, right = st.columns([1, 1.2], gap="large", vertical_alignment="center")
    with left:
        resolution_frame = pd.DataFrame(
            [
                ["224 × 224", "50,176", 0.791],
                ["128 × 128", "16,384", 0.827],
                ["16 × 16", "256", 0.835],
            ],
            columns=["Input", "Pixels", "Macro F1"],
        )
        st.dataframe(resolution_frame, hide_index=True, width="stretch")
    with right:
        st.write(
            "The simple CNN was trained at three resolutions. At 16 × 16 the lungs cover roughly "
            "8 × 8 pixels—far too few to show opacity or consolidation. Performance does not drop."
        )

    st.markdown("#### 16 × 16, three regions")
    full_col, bg_col, lung_col = st.columns(3, gap="small")
    full_col.metric("Full image", "0.835", border=True)
    bg_col.metric("Background only", "0.787", "-0.048", border=True)
    lung_col.metric("Lungs only", "0.749", "-0.086", border=True)
    st.write(
        "A 23,556-parameter network, seeing a 16 × 16 thumbnail with the lungs deleted, still "
        "reaches 0.787 macro F1 (chance: 0.163). COVID is the clearest case: F1 **0.669** from "
        "the background, only **0.532** from the lungs."
    )

with limits:
    st.markdown("#### What the evidence supports")
    st.markdown(
        "- The headline score is real on this test split.\n"
        "- A substantial part of it is available from non-lung, source-linked cues.\n"
        "- The lungs still carry some signal—lungs-only models stay well above chance.\n"
    )
    st.markdown("#### Limitations of this study")
    limitations = [
        (
            "No external validation",
            "All results come from one dataset. Testing on hospitals not represented here is the "
            "decisive next step—we predict COVID recall would drop the most.",
        ),
        (
            "One run per condition",
            "Each configuration was trained once with a fixed seed, so small differences are indicative only.",
        ),
        (
            "Masks come from a model",
            "Segmentation errors propagate into every lungs-only and background-only experiment.",
        ),
        (
            "Hardware limits",
            "Full fine-tuning at 224 × 224 exhausted memory, so the region ablation uses frozen backbones.",
        ),
    ]
    for title, body in limitations:
        with st.container(border=True):
            st.markdown(f"**{title}** — {body}")

disclaimer()
footer("Trust and limitations")