from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from covid_xray.preprocessing.config import SplitConfig
from covid_xray.preprocessing.manifest import split_manifest

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
    align_image_to_mask_frame,
)
from covid_xray.preprocessing.transforms import read_grayscale
from covid_xray.training.data import build_raw_manifest
from covid_xray.cnn.dataset import load_masked_image

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

    output_dir = Path("reports/cnn/lung_roi_by_class")

    manifest = build_raw_manifest(
        raw_dir=args.raw_dir,
        class_folders=CLASS_FOLDERS,
        redundant_csv=args.redundant_csv,
    )

    splits = split_manifest(
        manifest,
        SplitConfig(
            val_size=0.15,
            test_size=0.15,
            random_state=args.seed,
        ),
    )

    test_frame = splits.test

    build_all_class_roi_figures(
        frame=test_frame,   # or val_frame / train_frame
        output_dir=output_dir,
        image_size=(128, 128),
        mask_threshold=127,
        samples_per_class=3,
        random_state=42,
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

CLASS_ORDER = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
]


def _safe_name(label: str) -> str:
    return label.lower().replace(" ", "_")


def _load_raw_grayscale(path: str) -> np.ndarray:
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)
    return tf.squeeze(image).numpy().astype(np.float32)


def _load_mask(mask_path: str) -> np.ndarray:
    raw = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(raw, channels=1)
    return tf.squeeze(mask).numpy().astype(np.uint8)


def _make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35):
    """
    Return an RGB image where the mask boundary is shown on top of the X-ray.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.contour(mask > 127, levels=[0.5], colors=["purple"], linewidths=1.0)
    ax.axis("off")
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    overlay = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return overlay


def build_single_class_roi_figure(
    class_name: str,
    frame: pd.DataFrame,
    output_dir: Path,
    image_size=(128, 128),
    mask_threshold: int = 127,
    samples_per_class: int = 3,
    random_state: int = 42,
):
    """
    Create one figure for one class:
    columns = [original x-ray, mask used to locate ROI, final lung ROI]
    """
    class_frame = frame[frame["class"] == class_name].copy()

    if class_frame.empty:
        print(f"No samples found for class: {class_name}")
        return None

    n = min(samples_per_class, len(class_frame))
    class_frame = class_frame.sample(n=n, random_state=random_state).reset_index(drop=True)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(9, 3 * n),
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in class_frame.iterrows():
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        # Raw original image
        image = _load_raw_grayscale(image_path)

        # Raw mask
        mask = _load_mask(mask_path)

        # Align image into mask frame so overlay is correct
        aligned_image, aligned_mask = align_image_to_mask_frame(image, mask)

        # Final lung ROI exactly as pipeline uses it
        lung_roi = load_masked_image(
            tf.constant(image_path),
            tf.constant(mask_path),
            image_size,
            region="lung_roi",
            mask_threshold=mask_threshold,
        )
        lung_roi = tf.squeeze(lung_roi).numpy()

        # Column 1: original
        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].set_title(f"{class_name}\nOriginal X-ray", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Column 2: aligned image + mask outline
        axes[row_idx, 1].imshow(aligned_image, cmap="gray")
        axes[row_idx, 1].contour(
            aligned_mask > mask_threshold,
            levels=[0.5],
            colors=["purple"],
            linewidths=1.0,
        )
        axes[row_idx, 1].set_title("Mask used only to locate ROI", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Column 3: final ROI
        axes[row_idx, 2].imshow(lung_roi, cmap="gray")
        axes[row_idx, 2].set_title(
            f"Final lung ROI\n{image_size[0]}x{image_size[1]}",
            fontsize=10,
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle(f"Lung ROI preprocessing — {class_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lung_roi_{_safe_name(class_name)}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path


def build_all_class_roi_figures(
    frame: pd.DataFrame,
    output_dir: Path,
    image_size=(128, 128),
    mask_threshold: int = 127,
    samples_per_class: int = 3,
    random_state: int = 42,
):
    saved_paths = []

    for class_name in CLASS_ORDER:
        path = build_single_class_roi_figure(
            class_name=class_name,
            frame=frame,
            output_dir=output_dir,
            image_size=image_size,
            mask_threshold=mask_threshold,
            samples_per_class=samples_per_class,
            random_state=random_state,
        )
        if path is not None:
            saved_paths.append(path)

    return saved_paths

if __name__ == "__main__":
    main()
