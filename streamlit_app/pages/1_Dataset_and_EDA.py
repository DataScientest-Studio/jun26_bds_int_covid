import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lib.paths import LATEX_FIGURES_DIR
from lib.ui import app_footer, configure_page, narrative_row, page_header, safe_image

configure_page("Dataset & EDA")

page_header(
    "Dataset & EDA",
    "21,165 chest X-rays stitched together from different source repositories per class, "
    "a known recipe for hidden shortcuts.",
)

narrative_row(
    challenge=(
        "The four classes come from different source repositories (RSNA, Kaggle Pneumonia, "
        "SIRM/GitHub/Twitter for COVID), not one unified acquisition process."
    ),
    did=(
        "Audited structural integrity, class balance (chi-square test), RGB-encoding, exact "
        "duplicates (pHash), and brightness/contrast (Kruskal-Wallis, Cohen's d)."
    ),
    found=(
        "140 stray RGB images only in Viral Pneumonia, 59 duplicates concentrated in COVID, "
        "confirmed class imbalance, and early hints of the shortcut-learning risk proven "
        "causally in Step 6."
    ),
)

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Chest X-rays", "21,165", "21,106 after dedup")
metric_cols[1].metric("Diagnostic classes", "4")
metric_cols[2].metric("Stray RGB images", "140", "Viral Pneumonia only")
metric_cols[3].metric("Exact duplicates", "59", "0.28% of dataset")

balance_tab, integrity_tab, duplicates_tab, intensity_tab = st.tabs(
    ["Class Balance", "Integrity & Encoding", "Duplicates", "Brightness & Contrast"]
)

with balance_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(LATEX_FIGURES_DIR / "class_distribution.png", width="stretch")
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": ["Normal", "Lung Opacity", "COVID-19", "Viral Pneumonia"],
                    "Images": [10192, 6012, 3616, 1345],
                    "Share": ["48.15%", "28.41%", "17.08%", "6.35%"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        st.caption(
            "Chi-square goodness-of-fit: \u03c7\u00b2(3) = 8110.78, p < 0.001, Cohen's w = "
            "0.619 \u2014 large, significant imbalance. Requires stratified 70/15/15 "
            "splitting."
        )

with integrity_tab:
    st.caption(
        "Every one of the 21,165 X-rays has a matching segmentation mask by filename: 0 "
        "unreadable files, 0 unmatched pairs. X-rays are 299\u00d7299 px, masks are "
        "256\u00d7256 px, so pixel-wise mask operations need resizing or alignment."
    )
    left, right = st.columns([2, 3], gap="medium")
    with left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Mode": ["L (grayscale)", "RGB"],
                    "n": [1205, 140],
                    "Mean size (KB)": [37.45, 55.82],
                    "Median (KB)": [37.47, 56.28],
                }
            ),
            hide_index=True,
            width="stretch",
            height=110,
        )
        st.caption(
            "All 140 RGB images sit inside Viral Pneumonia (10.41% of that class); every "
            "other class is 100% grayscale. Encoding is confounded with diagnostic label."
        )
    with right:
        safe_image(LATEX_FIGURES_DIR / "viral_pneumonia_file_size.png", width="stretch")

with duplicates_tab:
    left, right = st.columns([2, 3], gap="medium")
    with left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": ["COVID-19", "Viral Pneumonia", "Normal", "Lung Opacity"],
                    "Redundant files": [51, 7, 1, 0],
                    "Share of class": ["1.41%", "0.52%", "0.01%", "0.00%"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        st.caption(
            "pHash screening flagged 287 candidate pairs; 91 were exact pixel duplicates. "
            "Grouping connected duplicate sets left 59 redundant files, all removed before "
            "splitting to avoid train/test leakage."
        )
    with right:
        safe_image(LATEX_FIGURES_DIR / "duplicate_distribution.png", width="stretch")

with intensity_tab:
    left, middle, right = st.columns([2, 2, 2], gap="medium")
    with left:
        safe_image(LATEX_FIGURES_DIR / "brightness_contrast_boxplots.png", width="stretch")
    with middle:
        safe_image(LATEX_FIGURES_DIR / "brightness_contrast_scatter.png", width="stretch")
    with right:
        st.caption(
            "Kruskal-Wallis: brightness H(3) = 987.39, contrast H(3) = 1393.78 (both p < "
            "0.001), effect sizes small-to-modest (\u03b5\u00b2 = 0.047 / 0.066). Largest "
            "pairwise gap: COVID-19 vs Normal contrast, Cohen's d = -0.618. Overlap across "
            "classes is substantial, so intensity alone will not separate them."
        )

app_footer()
