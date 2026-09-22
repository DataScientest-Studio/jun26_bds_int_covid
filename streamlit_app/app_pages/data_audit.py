from pathlib import Path

import pandas as pd
import streamlit as st

from lib.paths import FIGURES_DIR, example_image
from lib.ui import figure, footer, narrative, page_intro


def mask_for(image_path: Path) -> Path | None:
    """Locate the lung mask that belongs to an example image.

    Checks a ``masks/`` folder next to the example first (the bundled-assets
    layout), then the raw dataset layout ``<class>/masks/``, and returns None if
    neither exists so a missing file never breaks the page.
    """
    image_path = Path(image_path)
    for candidate in (
        image_path.parent / "masks" / image_path.name,
        image_path.parent.parent / "masks" / image_path.name,
    ):
        if candidate.exists():
            return candidate
    return None


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

st.markdown("#### One representative X-ray per class, with its lung mask")
examples = [
    ("COVID", example_image("COVID", "COVID-1.png")),
    ("Lung Opacity", example_image("Lung_Opacity", "Lung_Opacity-1.png")),
    ("Normal", example_image("Normal", "Normal-1.png")),
    ("Viral Pneumonia", example_image("Viral Pneumonia", "Viral Pneumonia-1.png")),
]
for column, (label, path) in zip(st.columns(4, gap="small"), examples):
    with column.container(border=True, height="stretch"):
        figure(path, label)
        mask_path = mask_for(path)
        if mask_path is not None:
            figure(mask_path, "Lung mask")
        else:
            st.caption("Mask not available")

st.caption(
    "Every image comes with a lung mask (256 × 256). The masks were produced by a "
    "segmentation model, so they are a useful tool rather than a neutral ground truth. "
    "Later experiments use them to restrict a model to the lungs, or to everything except the lungs."
)

overview, evidence = st.tabs(["Distribution & sources", "Audit findings"])

with overview:
    left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with left:
        figure(
            FIGURES_DIR / "class_distribution.png",
            "Normal is 48.2% of the raw collection; Viral Pneumonia is 6.4%.",
        )
        with st.container(horizontal=True):
            st.metric("Raw images", "21,165", border=True)
            st.metric("After deduplication", "21,106", border=True)
            st.metric("Image size", "299 × 299", border=True)
    with right:
        st.markdown("#### Source and label move together")
        source_frame = pd.DataFrame(
            [
                ["COVID", "Six public repositories", "3,616", "Exclusive — no other class"],
                ["Lung Opacity", "RSNA", "6,012", "Shared with Normal"],
                ["Normal", "RSNA (8,851) + Kaggle (1,341)", "10,192", "Shared"],
                ["Viral Pneumonia", "Kaggle (paediatric)", "1,345", "Shared with Normal"],
            ],
            columns=["Class", "Source repositories", "Images", "Source overlap"],
        )
        st.dataframe(source_frame, hide_index=True)
        with st.container(horizontal=True):
            st.metric("COVID F1 from source alone", "1.00", border=True)
            st.metric("Accuracy from source alone", "65.3%", border=True)
        st.warning(
            "Knowing only which repository an image came from—without looking at a single "
            "pixel—identifies every COVID case. Any source signature visible in the image "
            "can therefore stand in for the diagnosis.",
            icon=":material/warning:",
        )

with evidence:
    st.markdown("#### Finding → action")
    actions = pd.DataFrame(
        [
            ["Source–label link", "All COVID images come from COVID-only repositories", "Region and resolution ablations"],
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

st.info(
    "Audit consequence: evaluate macro F1 and per-class recall, then test where predictive information comes from.",
    icon=":material/arrow_forward:",
)

footer("Data audit")