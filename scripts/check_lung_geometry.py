from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covid_xray.cnn.lung_normalization import (
    align_image_to_mask_frame,
    estimate_lung_rotation,
    estimate_lung_rotation_raw,
    normalize_lung_geometry_pair,
)

from covid_xray.preprocessing.transforms import read_grayscale
from covid_xray.training.data import build_raw_manifest
from covid_xray.config import (
    CLASS_COLUMN,
    IMAGE_PATH_COLUMN,
    MASK_PATH_COLUMN,
)
CSV_PATH = Path("YOUR_MANIFEST_OR_TEST_CSV.csv")
OUTPUT_PATH = Path("reports/cnn/geometry_normalization_check.png")

N_IMAGES = 20
IMAGE_SIZE = (128, 128)
MASK_THRESHOLD = 127

manifest = build_raw_manifest()

print(manifest.columns)
print(manifest.head())

sample = manifest.sample(
    n=min(N_IMAGES, len(manifest)),
    random_state=42,
)
fig, axes = plt.subplots(
    len(sample),
    4,
    figsize=(12, 3 * len(sample)),
    squeeze=False,
)

for row_idx, (_, row) in enumerate(sample.iterrows()):

    image = read_grayscale(
        str(row[IMAGE_PATH_COLUMN])
    )

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )
    aligned_image, aligned_mask = align_image_to_mask_frame(
        image,
        mask,
    )
    normalized, normalized_mask, angle = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=IMAGE_SIZE,
            mask_threshold=MASK_THRESHOLD,
        )
    )

    class_name = row[CLASS_COLUMN]

    # Original image
    axes[row_idx, 0].imshow(
        image,
        cmap="gray",
    )
    axes[row_idx, 0].set_title(
        f"{class_name}\nOriginal"
    )

    axes[row_idx, 1].imshow(
        aligned_image,
        cmap="gray",
    )

    axes[row_idx, 1].contour(
        aligned_mask > MASK_THRESHOLD,
        levels=[0.5],
    )

    axes[row_idx, 1].set_title(
        f"Aligned image + mask\nrotation={angle:.1f}°"
    )

    # Geometry-normalized image
    axes[row_idx, 2].imshow(
        normalized,
        cmap="gray",
    )

    axes[row_idx, 2].set_title(
        "Geometry normalized"
    )

    # Geometry-normalized mask overlay
    axes[row_idx, 3].imshow(
        normalized,
        cmap="gray",
    )

    axes[row_idx, 3].contour(
        normalized_mask > 0,
        levels=[0.5],
    )

    axes[row_idx, 3].set_title(
        "Normalized mask"
    )

    for axis in axes[row_idx]:
        axis.axis("off")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved QA figure to: {OUTPUT_PATH}")

angles = []

for _, row in manifest.iterrows():

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    binary = (
        mask > MASK_THRESHOLD
    ).astype(np.uint8)

    angle = estimate_lung_rotation(
        binary
    )

    angles.append(angle)

angles = np.asarray(angles)

print("\n=== ROTATION STATISTICS ===")
print(f"Mean:   {angles.mean():.3f}")
print(f"Median: {np.median(angles):.3f}")
print(f"Min:    {angles.min():.3f}")
print(f"Max:    {angles.max():.3f}")

print(
    "Fraction hitting ±10° clipping:",
    np.mean(np.abs(angles) >= 9.99),
)

component_counts = []

for _, row in manifest.iterrows():

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    binary = (
        mask > MASK_THRESHOLD
    ).astype(np.uint8)

    n_labels, _, _, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    component_counts.append(
        n_labels - 1
    )

print("\n=== MASK COMPONENT COUNTS ===")

print(
    pd.Series(component_counts)
    .value_counts()
    .sort_index()
)

print(
    "Fraction at rotation limit:",
    np.mean(np.abs(angles) >= 9.99)
)

normalized_area_fractions = []

for _, row in manifest.iterrows():

    image = read_grayscale(
        str(row[IMAGE_PATH_COLUMN])
    )

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    _, normalized_mask, _ = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=(128, 128),
            mask_threshold=127,
        )
    )

    normalized_area_fractions.append(
        np.mean(normalized_mask > 0)
    )

normalized_area_fractions = np.asarray(
    normalized_area_fractions
)

print("\n=== NORMALIZED LUNG AREA ===")
print("Mean:", normalized_area_fractions.mean())
print("Std:", normalized_area_fractions.std())
print("Min:", normalized_area_fractions.min())
print("Max:", normalized_area_fractions.max())

before = []
after = []

for _, row in manifest.iterrows():

    image = read_grayscale(
        str(row[IMAGE_PATH_COLUMN])
    )

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    aligned_image, aligned_mask = align_image_to_mask_frame(
        image,
        mask,
    )

    before.append(
        np.mean(aligned_mask > 127)
    )

    _, normalized_mask, _ = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=(128, 128),
            mask_threshold=127,
        )
    )

    after.append(
        np.mean(normalized_mask > 0)
    )

print("Before area std:", np.std(before))
print("After area std: ", np.std(after))

raw_angles = []
clipped_angles = []

for _, row in manifest.iterrows():
    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    binary = (
        mask > MASK_THRESHOLD
    ).astype(np.uint8)

    raw_angles.append(
        estimate_lung_rotation_raw(binary)
    )

    clipped_angles.append(
        estimate_lung_rotation(binary)
    )

raw_angles = np.asarray(raw_angles)
clipped_angles = np.asarray(clipped_angles)

print("\n=== RAW ROTATION STATISTICS ===")
print(f"Mean:   {raw_angles.mean():.3f}")
print(f"Median: {np.median(raw_angles):.3f}")
print(f"Min:    {raw_angles.min():.3f}")
print(f"Max:    {raw_angles.max():.3f}")

print(
    "Fraction requiring >10° correction:",
    np.mean(np.abs(raw_angles) > 10.0),
)

print(
    "Fraction requiring >15° correction:",
    np.mean(np.abs(raw_angles) > 15.0),
)

print(
    "Fraction requiring >20° correction:",
    np.mean(np.abs(raw_angles) > 20.0),
)

def mask_bbox_fraction(
    mask: np.ndarray,
) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.nan, np.nan

    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1

    return (
        bbox_h / mask.shape[0],
        bbox_w / mask.shape[1],
    )

bbox_heights = []
bbox_widths = []

for _, row in manifest.iterrows():

    image = read_grayscale(
        str(row[IMAGE_PATH_COLUMN])
    )

    mask = read_grayscale(
        str(row[MASK_PATH_COLUMN])
    )

    _, normalized_mask, _ = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=(128, 128),
            mask_threshold=MASK_THRESHOLD,
        )
    )

    h_fraction, w_fraction = (
        mask_bbox_fraction(normalized_mask)
    )

    bbox_heights.append(h_fraction)
    bbox_widths.append(w_fraction)

print("\n=== NORMALIZED BOUNDING BOX ===")

print(
    "Height mean/std:",
    np.nanmean(bbox_heights),
    np.nanstd(bbox_heights),
)

print(
    "Width mean/std:",
    np.nanmean(bbox_widths),
    np.nanstd(bbox_widths),
)

