import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st

from lib.paths import DATA_DIR, LATEX_FIGURES_DIR

PREPROCESSING_JSON = DATA_DIR / "arrays" / "preprocessing.json"
from lib.ui import app_footer, configure_page, key_finding, narrative_row, page_header, safe_image

configure_page("Preprocessing")

page_header(
    "Preprocessing",
    "Fix what EDA exposed without introducing new bias, and make every downstream "
    "experiment reproducible from the same stratified split.",
)

narrative_row(
    challenge=(
        "EDA surfaced duplicates, class-specific RGB encoding, mismatched mask "
        "resolutions, imbalance, and global intensity differences. Every fix had "
        "to avoid new shortcuts and stay auditable."
    ),
    did=(
        "Removed 59 exact duplicates before splitting, applied a stratified "
        "70/15/15 partition (seed 42), standardized to grayscale, aligned masks "
        "with nearest-neighbor resizing, normalized each image independently, and "
        "packaged a mask-safe augmentation pipeline for later ablations."
    ),
    found=(
        "21,106 modeling-ready images with split integrity verified across 78 "
        "automated tests, including a bit-identical full-pipeline rebuild check. "
        "Reproducibility was treated as a first-class requirement, not an "
        "afterthought."
    ),
)

preprocessing_meta = None
if PREPROCESSING_JSON.exists():
    try:
        with open(PREPROCESSING_JSON, "r") as handle:
            preprocessing_meta = json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        preprocessing_meta = None

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("After deduplication", "21,106", "59 removed pre-split")
metric_cols[1].metric("Split ratio", "70/15/15", "stratified, seed 42")
metric_cols[2].metric("Output size", "224×224", "grayscale baseline")
metric_cols[3].metric("Automated checks", "78", "transforms + pipeline")

workflow_tab, split_tab, transforms_tab, augmentation_tab, verification_tab = st.tabs(
    ["Workflow", "Stratified Split", "Per-Image Pipeline", "Augmentation", "Verification"]
)

with workflow_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(LATEX_FIGURES_DIR / "preprocessing_transform_steps.png", width="stretch")
    with right:
        st.caption(
            "Raw dataset (21,165) → remove 59 exact duplicates → cleaned set "
            "(21,106) → stratified 70/15/15 split → per-image pipeline "
            "(grayscale, resize, optional CLAHE / lung mask / augmentation, "
            "min–max normalization) → train, validation, and test partitions."
        )
        key_finding(
            "Duplicates were removed before splitting so identical observations "
            "cannot leak across train and test."
        )

with split_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        if preprocessing_meta and "class_counts" in preprocessing_meta:
            rows = []
            for class_name in ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]:
                train_count = preprocessing_meta["class_counts"]["train"].get(class_name, 0)
                val_count = preprocessing_meta["class_counts"]["val"].get(class_name, 0)
                test_count = preprocessing_meta["class_counts"]["test"].get(class_name, 0)
                rows.append(
                    {
                        "Class": class_name,
                        "Train": train_count,
                        "Validation": val_count,
                        "Test": test_count,
                        "Total": train_count + val_count + test_count,
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=180)
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Class": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
                        "Train": [2495, 4208, 7134, 937],
                        "Validation": [535, 902, 1528, 201],
                        "Test": [535, 902, 1529, 200],
                        "Total": [3565, 6012, 10191, 1338],
                    }
                ),
                hide_index=True,
                width="stretch",
                height=180,
            )
    with right:
        st.caption(
            "Two-step stratified procedure: 70% train and 30% holdout, then the "
            "holdout split equally into validation and test. Class proportions stay "
            "aligned within 0.05 percentage points across splits."
        )
        if preprocessing_meta and "samples" in preprocessing_meta:
            samples = preprocessing_meta["samples"]
            st.caption(
                f"Recorded totals: {samples.get('train', 14774):,} train + "
                f"{samples.get('val', 3166):,} val + "
                f"{samples.get('test', 3166):,} test = "
                f"{sum(samples.values()):,} images."
            )

with transforms_tab:
    left, middle, right = st.columns([2, 2, 2], gap="medium")
    with left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Setting": [
                        "Grayscale conversion",
                        "Resize (image)",
                        "Resize (mask)",
                        "CLAHE",
                        "Lung masking",
                        "Min–max normalization",
                    ],
                    "Baseline": [
                        "On",
                        "224×224, area",
                        "224×224, nearest",
                        "Off",
                        "Off",
                        "Per image, ε=1e-8",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
            height=220,
        )
    with middle:
        st.caption(
            "Grayscale removes the Viral Pneumonia RGB confound. Nearest-neighbor "
            "mask resizing preserves binary lung boundaries when aligning 256×256 "
            "masks with 299×299 X-rays."
        )
        st.caption(
            "Per-image min–max normalization attenuates scanner-specific global "
            "brightness differences flagged in EDA without assuming a fixed "
            "dataset-wide intensity scale."
        )
    with right:
        key_finding(
            "CLAHE, lung masking, and augmentation stay available as controlled "
            "ablations rather than being baked into the baseline, so their effect "
            "can be measured downstream."
        )

with augmentation_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        safe_image(LATEX_FIGURES_DIR / "preprocessing_augmentation_ops.png", width="stretch")
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Operation": [
                        "Horizontal flip",
                        "Rotation + zoom compensation",
                        "Random zoom",
                        "Translation",
                        "Brightness / contrast",
                    ],
                    "Probability": ["0.5", "0.5", "0.5", "0.5", "0.4"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
        st.caption(
            "Geometric transforms apply to image and mask jointly; photometric "
            "changes affect the image only. Augmentation is restricted to the "
            "training split."
        )
        safe_image(LATEX_FIGURES_DIR / "preprocessing_augment_pair.png", width="stretch")

with verification_tab:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Check": [
                        "Transform correctness",
                        "Split integrity",
                        "End-to-end consistency",
                        "Full pipeline rebuild",
                    ],
                    "Result": [
                        "Normalized outputs in [0, 1]; masking zeroes background",
                        "Disjoint splits; proportions preserved",
                        "21,106 images read with zero failures",
                        "Bit-identical to saved arrays",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
            height=180,
        )
    with right:
        key_finding(
            "78 automated tests cover transforms, augmentation, splitting, and "
            "pipeline regression. Stochastic steps use random seed 42 throughout."
        )
        st.caption(
            "Viral Pneumonia remains 6.34% of the training set after splitting, "
            "so downstream evaluation must report per-class precision, recall, "
            "macro F1, and confusion matrices rather than accuracy alone."
        )

app_footer()
