import pandas as pd
import streamlit as st

from lib.paths import DATA_DIR, FIGURES_DIR
from lib.ui import figure, footer, narrative, page_intro


page_intro(
    "01 · Data audit",
    "The dataset is useful—and structurally confounded",
    "We audited what the model could learn before asking how accurately it could learn it.",
)

narrative(
    "Separate disease signal from class frequency, file properties, acquisition source, and mask geometry.",
    "Inspected representative scans, class/source composition, encoding, duplicates, intensity, and masks.",
    "Class and source are entangled, so a strong internal test score can still rely on shortcuts.",
)

st.markdown("#### One representative X-ray per class")
examples = [
    ("COVID", DATA_DIR / "processed" / "COVID" / "images" / "COVID-1.png"),
    ("Lung Opacity", DATA_DIR / "processed" / "Lung_Opacity" / "images" / "Lung_Opacity-1.png"),
    ("Normal", DATA_DIR / "processed" / "Normal" / "images" / "Normal-1.png"),
    ("Viral Pneumonia", DATA_DIR / "processed" / "Viral Pneumonia" / "images" / "Viral Pneumonia-1.png"),
]
for column, (label, path) in zip(st.columns(4, gap="small"), examples):
    with column.container(border=True, height="stretch"):
        st.image(str(path))
        st.markdown(f"**{label}**")

overview, evidence, geometry = st.tabs(
    ["Distribution & sources", "Three audit findings", "Supporting mask evidence"]
)

with overview:
    left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "class_distribution.png",
            "Normal is 48.2% of the raw collection; Viral Pneumonia is 6.4%.",
        )
    with right:
        st.markdown("#### Source and label move together")
        source_frame = pd.DataFrame(
            [
                ["COVID", "Multiple public repositories", "Mixed"],
                ["Lung Opacity", "RSNA only", "Single source"],
                ["Normal", "Mainly RSNA + Kaggle", "Mixed"],
                ["Viral Pneumonia", "Kaggle only", "Single source"],
            ],
            columns=["Class", "Documented origin", "Source pattern"],
        )
        st.dataframe(source_frame, hide_index=True)
        st.warning(
            "A source-specific border, scanner, or preprocessing signature can become a proxy for the diagnosis.",
            icon=":material/warning:",
        )
        with st.container(horizontal=True):
            st.metric("Raw images", "21,165", border=True)
            st.metric("Image size", "299 × 299", border=True)
            st.metric("Classes", "4", border=True)

with evidence:
    st.markdown("#### Finding → action")
    actions = pd.DataFrame(
        [
            ["RGB encoding", "140 RGB files, all Viral Pneumonia", "Convert every image to grayscale"],
            ["Exact duplicates", "59 redundant files, concentrated in COVID", "Deduplicate before splitting"],
            ["Intensity differences", "Brightness and contrast vary by class", "Test lung-constrained variants"],
        ],
        columns=["Finding", "Evidence", "Action"],
    )
    st.dataframe(actions, hide_index=True)

    encoding, duplicates, intensity = st.tabs(["RGB encoding", "Duplicates", "Intensity"])
    with encoding:
        figure(
            FIGURES_DIR / "viral_pneumonia_file_size.png",
            "All 140 RGB files belong to Viral Pneumonia and are larger than grayscale files.",
        )
    with duplicates:
        figure(
            FIGURES_DIR / "duplicate_distribution.png",
            "Removing 59 redundant files before the split protects the integrity of evaluation.",
        )
    with intensity:
        figure(
            FIGURES_DIR / "brightness_contrast_boxplots.png",
            "Whole-image brightness and contrast differ systematically across classes.",
        )

with geometry:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "mean_lung_mask_area_ratio.png",
            "Average mask coverage differs by class.",
        )
    with right:
        st.markdown("#### Supporting evidence—not a separate claim")
        st.write(
            "The supplied masks are not automatically neutral: their area and shape can retain source or annotation conventions. We therefore used them as controlled experimental inputs, not as guaranteed bias removal."
        )
        st.info(
            "Audit consequence: evaluate macro F1 and per-class recall, then test where predictive information comes from.",
            icon=":material/arrow_forward:",
        )

footer("Data audit")
