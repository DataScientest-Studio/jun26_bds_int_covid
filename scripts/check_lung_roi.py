from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covid_xray.config import (
    CLASS_COLUMN,
    CLASS_FOLDERS,
    IMAGE_PATH_COLUMN,
    MASK_PATH_COLUMN,
    RAW_DIR,
)
from covid_xray.cnn.lung_normalization import (
    extract_lung_roi,
    resize_mask_to_image_frame,
)
from covid_xray.preprocessing.transforms import read_grayscale
from covid_xray.training.data import build_raw_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual QA for the lung-ROI baseline preprocessing."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--redundant-csv", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cnn/lung_roi_preprocessing_check.png"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest = build_raw_manifest(
        raw_dir=args.raw_dir,
        class_folders=CLASS_FOLDERS,
        redundant_csv=args.redundant_csv,
    )

    sampled = []
    for _, group in manifest.groupby(CLASS_COLUMN):
        sampled.append(
            group.sample(
                n=min(args.samples_per_class, len(group)),
                random_state=args.seed,
            )
        )

    sample = (
        pd.concat(sampled, ignore_index=True)
        .sort_values(CLASS_COLUMN)
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        len(sample),
        3,
        figsize=(10, 3 * len(sample)),
        squeeze=False,
    )

    for row_idx, (_, row) in enumerate(sample.iterrows()):
        image = read_grayscale(str(row[IMAGE_PATH_COLUMN]))
        mask = read_grayscale(str(row[MASK_PATH_COLUMN]))

        aligned_mask = resize_mask_to_image_frame(
            mask,
            np.squeeze(image).shape,
        )

        roi = extract_lung_roi(
            image=image,
            mask=mask,
            target_size=tuple(args.image_size),
            mask_threshold=args.mask_threshold,
        )

        class_name = str(row[CLASS_COLUMN])

        axes[row_idx, 0].imshow(np.squeeze(image), cmap="gray")
        axes[row_idx, 0].set_title(f"{class_name}\nOriginal X-ray")

        axes[row_idx, 1].imshow(np.squeeze(image), cmap="gray")
        axes[row_idx, 1].contour(
            aligned_mask > args.mask_threshold,
            levels=[0.5],
        )
        axes[row_idx, 1].set_title("Mask used only to locate ROI")

        axes[row_idx, 2].imshow(np.squeeze(roi), cmap="gray")
        axes[row_idx, 2].set_title(
            f"Final lung ROI\n{args.image_size[0]}×{args.image_size[1]}"
        )

        for axis in axes[row_idx]:
            axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        args.output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved lung-ROI QA figure to: {args.output}")


if __name__ == "__main__":
    main()
