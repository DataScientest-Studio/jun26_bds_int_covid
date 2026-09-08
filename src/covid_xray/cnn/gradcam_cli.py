from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from ..config import (
    CLASS_COLUMN,
    CLASS_FOLDERS,
    IMAGE_PATH_COLUMN,
    MASK_PATH_COLUMN,
    RANDOM_STATE,
    RAW_DIR,
)
from .config import CNN_REGIONS
from ..explainability.gradcam import gradcam_for_image
from ..preprocessing.config import SplitConfig
from ..preprocessing.manifest import split_manifest
from ..preprocessing.transforms import (
    read_grayscale,
    resize_mask,
)
from ..training.data import build_raw_manifest
from .dataset import load_masked_image
from ..cnn.lung_normalization import (
    extract_lung_roi_mask,
)

# ---------------------------------------------------------
# Load image exactly as CNN saw it during training
# ---------------------------------------------------------

def load_cnn_gradcam_input(
    image_path: str,
    mask_path: str,
    image_size: tuple[int, int],
    region: str,
    mask_threshold: int,
) -> np.ndarray:
    """Load one CNN input using the same preprocessing as training."""

    image = load_masked_image(
        tf.constant(image_path),
        tf.constant(mask_path),
        image_size,
        region,
        mask_threshold,
    )

    return image.numpy()


# ---------------------------------------------------------
# Recreate test split
# ---------------------------------------------------------

def build_test_frame(args):
    """Recreate the same deduplicated test split used for training."""

    manifest = build_raw_manifest(
        raw_dir=args.raw_dir,
        class_folders=CLASS_FOLDERS,
        redundant_csv=args.redundant_csv,
    )

    splits = split_manifest(
        manifest,
        SplitConfig(
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed,
        ),
    )

    return splits.test


# ---------------------------------------------------------
# Select examples
# ---------------------------------------------------------

def sample_test_images(
    frame,
    samples_per_class: int,
    random_state: int,
):
    sampled_frames = []

    for _, group in frame.groupby(CLASS_COLUMN):
        n = min(samples_per_class, len(group))

        sampled_frames.append(
            group.sample(
                n=n,
                random_state=random_state,
            )
        )

    return pd.concat(
        sampled_frames,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Generate Grad-CAM figure
# ---------------------------------------------------------

def generate_gradcam_figure(
    model: keras.Model,
    frame,
    output_path: Path,
    image_size: tuple[int, int],
    region: str,
    mask_threshold: int,
    samples_per_class: int,
    random_state: int,
    target_layer: str | None = None,
) -> Path:

    samples = sample_test_images(
        frame,
        samples_per_class=samples_per_class,
        random_state=random_state,
    )

    n_rows = len(samples)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10, 3 * n_rows),
        squeeze=False,
    )

    lung_fractions = []

    for row_index, (_, row) in enumerate(samples.iterrows()):

        image_path = str(row[IMAGE_PATH_COLUMN])
        mask_path = str(row[MASK_PATH_COLUMN])

        # What the CNN actually receives
        model_input = load_cnn_gradcam_input(
            image_path=image_path,
            mask_path=mask_path,
            image_size=image_size,
            region=region,
            mask_threshold=mask_threshold,
        )

        # Lung mask for attention measurement / contour
        mask = read_grayscale(mask_path)
        mask = resize_mask(
            mask,
            image_size,
        )
        lung_area_fraction = np.mean(
            mask > mask_threshold
        )
        # Grad-CAM
        result = gradcam_for_image(
            model=model,
            image_array=model_input,
            mask=mask,
            target_layer_name=target_layer,
            mask_threshold=mask_threshold,
        )

        true_label = row[CLASS_COLUMN]
        predicted_label = result["predicted_label"]
        lung_fraction = result["lung_fraction"]

        lung_fractions.append(
            {
                "true": true_label,
                "predicted": predicted_label,
                "lung_fraction": lung_fraction,
                "lung_area_fraction": lung_area_fraction,

            }
        )

        # ---------------------------------------------
        # Column 1: actual model input
        # ---------------------------------------------

        input_display = model_input[..., 0]

        axes[row_index, 0].imshow(
            input_display,
            cmap="gray",
        )

        axes[row_index, 0].set_title(
            f"Input ({region})\n"
            f"True: {true_label}"
        )

        # ---------------------------------------------
        # Column 2: Grad-CAM heatmap
        # ---------------------------------------------

        axes[row_index, 1].imshow(
            result["heatmap"],
            cmap="jet",
        )

        axes[row_index, 1].set_title(
            "Grad-CAM heatmap"
        )

        # ---------------------------------------------
        # Column 3: overlay
        # ---------------------------------------------

        axes[row_index, 2].imshow(
            result["overlay"]
        )

        fraction_text = (
            f"{lung_fraction:.2%}"
            if lung_fraction is not None
            and not np.isnan(lung_fraction)
            else "N/A"
        )

        axes[row_index, 2].set_title(
            f"Pred: {predicted_label}\n"
            f"Lung attention: {fraction_text}"
        )

        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(
        f"Grad-CAM — {model.name} — region={region}",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    attention_values = [
    item["lung_fraction"]
    for item in lung_fractions
    if item["lung_fraction"] is not None
    and not np.isnan(item["lung_fraction"])
    ]

    area_values = [
        item["lung_area_fraction"]
        for item in lung_fractions
    ]

    if attention_values and area_values:
        mean_attention = np.mean(attention_values)
        mean_area = np.mean(area_values)

        print()
        print(
            f"Mean lung attention fraction: {mean_attention:.3f}"
        )
        print(
            f"Mean lung area fraction:      {mean_area:.3f}"
        )
        print(
            f"Lung attention enrichment:    "
            f"{mean_attention / mean_area:.3f}"
        )

    print(
        f"Saved Grad-CAM figure to: {output_path}"
    )

    return output_path


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM visualizations "
            "for CNN classifiers."
        )
    )

    parser.add_argument(
        "--save-classwise",
        action="store_true",
        help=(
            "Save a separate Grad-CAM figure "
            "for each class."
        ),
    )

    parser.add_argument(
        "--classwise-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for class-specific "
            "Grad-CAM figures."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained .keras model.",
    )

    parser.add_argument(
        "--region",
        choices=CNN_REGIONS,
        default="full",
        help="Image region used during model training.",
    )

    parser.add_argument(
        "--redundant-csv",
        type=Path,
        required=True,
        help=(
            "CSV containing redundant images "
            "that must be excluded."
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("HEIGHT", "WIDTH"),
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=SplitConfig.val_size,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=SplitConfig.test_size,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )

    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help=(
            "Optional convolutional layer name. "
            "If omitted, Grad-CAM automatically "
            "uses the last Conv2D layer."
        ),
    )

    return parser

def evaluate_lung_attention(
    model,
    frame,
    image_size,
    region,
    mask_threshold,
    target_layer=None,
):
    records = []

    for index, (_, row) in enumerate(frame.iterrows(), start=1):

        image_path = str(row[IMAGE_PATH_COLUMN])
        mask_path = str(row[MASK_PATH_COLUMN])

        model_input = load_cnn_gradcam_input(
            image_path=image_path,
            mask_path=mask_path,
            image_size=image_size,
            region=region,
            mask_threshold=mask_threshold,
        )

        mask = read_grayscale(mask_path)

        if region == "lung_roi":
            original_image = read_grayscale(
                image_path
            )

            mask = extract_lung_roi_mask(
                mask=mask,
                image_shape=np.squeeze(
                    original_image
                ).shape,
                target_size=image_size,
                mask_threshold=mask_threshold,
            )

        else:
            mask = resize_mask(
                mask,
                image_size,
            )
        result = gradcam_for_image(
            model=model,
            image_array=model_input,
            mask=mask,
            target_layer_name=target_layer,
            mask_threshold=mask_threshold,
        )

        true_class = row[CLASS_COLUMN]
        predicted_class = result["predicted_label"]

        lung_attention = result["lung_fraction"]

        lung_area = np.mean(mask > mask_threshold)

        enrichment = (
            lung_attention / lung_area
            if lung_area > 0
            else np.nan
        )

        records.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "correct": true_class == predicted_class,
                "lung_attention_fraction": lung_attention,
                "lung_area_fraction": lung_area,
                "lung_attention_enrichment": enrichment,
            }
        )

        if index % 100 == 0:
            print(f"Processed {index}/{len(frame)} images")

    return pd.DataFrame(records)

def select_representative_examples(
    attention_df: pd.DataFrame,
    only_correct: bool = True,
) -> pd.DataFrame:

    df = attention_df.copy()

    df = df.dropna(
        subset=["lung_attention_enrichment"]
    )

    if only_correct:
        df = df[df["correct"] == True].copy()

    selected = []

    for class_name in df["true_class"].unique():

        group = df[
            df["true_class"] == class_name
        ].copy()

        print(
            f"Selecting representatives for {class_name}: "
            f"{len(group)} candidates"
        )

        if group.empty:
            continue

        values = group["lung_attention_enrichment"]

        # Representative low / middle / high:
        # 10th percentile, median, 90th percentile
        targets = {
            "low": values.quantile(0.10),
            "mid": values.quantile(0.50),
            "high": values.quantile(0.90),
        }

        used_indices = set()

        for representative_type, target_value in targets.items():

            candidates = group.loc[
                ~group.index.isin(used_indices)
            ].copy()

            if candidates.empty:
                continue

            distances = (
                candidates["lung_attention_enrichment"]
                - target_value
            ).abs()

            selected_idx = distances.idxmin()

            selected_row = group.loc[selected_idx].copy()

            selected_row["representative_type"] = (
                representative_type
            )

            selected.append(selected_row)

            used_indices.add(selected_idx)

    result = pd.DataFrame(selected).reset_index(drop=True)

    print("\nSelected representatives:")
    print(result["true_class"].value_counts())

    return result


def generate_representative_gradcam_figure(
    model: keras.Model,
    representatives: pd.DataFrame,
    output_path: Path,
    image_size: tuple[int, int],
    region: str,
    mask_threshold: int,
    target_layer: str | None = None,
) -> Path:

    n_rows = len(representatives)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10, 3 * n_rows),
        squeeze=False,
    )

    for row_index, (_, row) in enumerate(representatives.iterrows()):

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        model_input = load_cnn_gradcam_input(
            image_path=image_path,
            mask_path=mask_path,
            image_size=image_size,
            region=region,
            mask_threshold=mask_threshold,
        )

        mask = read_grayscale(mask_path)

        if region == "lung_roi":
            original_image = read_grayscale(
                image_path
            )

            mask = extract_lung_roi_mask(
                mask=mask,
                image_shape=np.squeeze(
                    original_image
                ).shape,
                target_size=image_size,
                mask_threshold=mask_threshold,
            )

        else:
            mask = resize_mask(
                mask,
                image_size,
            )
        result = gradcam_for_image(
            model=model,
            image_array=model_input,
            mask=mask,
            target_layer_name=target_layer,
            mask_threshold=mask_threshold,
        )

        rep_type = row["representative_type"]
        true_label = row["true_class"]
        pred_label = row["predicted_class"]
        enrichment = row["lung_attention_enrichment"]
        correct = row["correct"]

        # Column 1: model input
        axes[row_index, 0].imshow(
            model_input[..., 0],
            cmap="gray",
        )
        axes[row_index, 0].set_title(
            f"{true_label} ({rep_type})\nInput"
        )

        # Column 2: heatmap
        axes[row_index, 1].imshow(
            result["heatmap"],
            cmap="jet",
        )
        axes[row_index, 1].set_title("Grad-CAM")

        # Column 3: overlay
        axes[row_index, 2].imshow(
            result["overlay"]
        )
        axes[row_index, 2].set_title(
            f"Pred: {pred_label}\n"
            f"Correct: {correct}\n"
            f"Enrichment: {enrichment:.3f}"
        )

        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(
        f"Representative Grad-CAMs — {model.name} — region={region}",
        fontsize=15,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved representative Grad-CAM figure to: {output_path}")
    return output_path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> int:

    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")

    model = keras.models.load_model(
        args.model_path
    )

    print(f"Model: {model.name}")
    print(f"Region: {args.region}")

    # Recreate exact deduplicated test split
    test_frame = build_test_frame(args)

    print(
        f"Test images available for Grad-CAM: "
        f"{len(test_frame)}"
    )

# ---------------------------------------------------------
# Optional: save one Grad-CAM figure per class
# ---------------------------------------------------------

    if args.save_classwise:

        classwise_output_dir = (
            args.classwise_output_dir
            if args.classwise_output_dir is not None
            else Path(
                f"reports/cnn/"
                f"{args.model_path.stem}_"
                f"{args.region}_gradcam_classwise"
            )
        )

        generate_classwise_gradcam_figures(
            model=model,
            frame=test_frame,
            output_dir=classwise_output_dir,
            image_size=tuple(args.image_size),
            region=args.region,
            mask_threshold=args.mask_threshold,
            samples_per_class=args.samples_per_class,
            random_state=args.seed,
            target_layer=args.target_layer,
        )

        print(
            f"\nSaved class-wise Grad-CAM figures to: "
            f"{classwise_output_dir}"
        )

        return 0
    # Default output path
    if args.output is None:

        suffix = (
            ""
            if args.region == "full"
            else f"_{args.region}"
        )

        args.output = Path(
            f"reports/cnn/"
            f"{args.model_path.stem}"
            f"{suffix}_gradcam.png"
        )

    generate_gradcam_figure(
        model=model,
        frame=test_frame,
        output_path=args.output,
        image_size=tuple(args.image_size),
        region=args.region,
        mask_threshold=args.mask_threshold,
        samples_per_class=args.samples_per_class,
        random_state=args.seed,
        target_layer=args.target_layer,
    )

    attention_df = evaluate_lung_attention(
        model=model,
        frame=test_frame,
        image_size=tuple(args.image_size),
        region=args.region,
        mask_threshold=args.mask_threshold,
        target_layer=args.target_layer,
    )   

    print("\n=== CORRECT VS INCORRECT GRAD-CAM ATTENTION ===")

    correctness_summary = (
        attention_df
        .groupby(["true_class", "correct"])
        .agg(
            n=("lung_attention_enrichment", "size"),
            mean_lung_attention=(
                "lung_attention_fraction",
                "mean",
            ),
            median_lung_attention=(
                "lung_attention_fraction",
                "median",
            ),
            mean_lung_area=(
                "lung_area_fraction",
                "mean",
            ),
            mean_enrichment=(
                "lung_attention_enrichment",
                "mean",
            ),
            median_enrichment=(
                "lung_attention_enrichment",
                "median",
            ),
        )
        .reset_index()
    )

    print(correctness_summary.to_string(index=False))

    metrics_path = Path(
        "reports/cnn/cnn_simple_gradcam_attention.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    attention_df.to_csv(
        metrics_path,
        index=False,
    )
    print()
    print("All-test Grad-CAM attention results")
    print(
        "Mean lung attention fraction:",
        attention_df["lung_attention_fraction"].mean(),
    )
    print(
        "Mean lung area fraction:",
        attention_df["lung_area_fraction"].mean(),
    )
    print(
        "Mean lung attention enrichment:",
        attention_df["lung_attention_enrichment"].mean(),
    )

    print()
    print(
        attention_df.groupby("true_class")[
            [
                "lung_attention_fraction",
                "lung_area_fraction",
                "lung_attention_enrichment",
            ]
        ].mean()
    )
    print("\n=== ALL ATTENTION ROWS ===")
    print(attention_df["true_class"].value_counts())

    print("\n=== CORRECT BY CLASS ===")
    print(
        attention_df[
            attention_df["correct"]
        ]["true_class"].value_counts()
    )

    print("\n=== VALID ENRICHMENT BY CLASS ===")
    print(
        attention_df.dropna(
            subset=["lung_attention_enrichment"]
        )["true_class"].value_counts()
    )

    print("\n=== CORRECT + VALID ENRICHMENT ===")
    print(
        attention_df[
            attention_df["correct"]
        ]
        .dropna(
            subset=["lung_attention_enrichment"]
        )["true_class"]
        .value_counts()
    )
    representatives = select_representative_examples(
        attention_df,
        only_correct=True,
    )

    representatives_path = Path(
        f"reports/cnn/{args.model_path.stem}_{args.region}_gradcam_representatives.csv"
    )
    representatives.to_csv(representatives_path, index=False)

    representative_figure_path = Path(
        f"reports/cnn/{args.model_path.stem}_{args.region}_gradcam_representatives.png"
    )
    gradcam_accuracy = attention_df["correct"].mean()

    print(
        f"Grad-CAM prediction accuracy: "
        f"{gradcam_accuracy:.4f}"
    )
    generate_representative_gradcam_figure(
        model=model,
        representatives=representatives,
        output_path=representative_figure_path,
        image_size=tuple(args.image_size),
        region=args.region,
        mask_threshold=args.mask_threshold,
        target_layer=args.target_layer,
    )
    return 0

def generate_classwise_gradcam_figures(
    model,
    frame: pd.DataFrame,
    output_dir: Path,
    image_size: tuple[int, int],
    region: str,
    mask_threshold: int,
    samples_per_class: int = 3,
    random_state: int = 42,
    target_layer: str | None = None,
) -> list[Path]:
    """
    Save one Grad-CAM figure per class.

    Each output figure contains:
        samples_per_class rows
        x
        3 columns:
            input
            Grad-CAM heatmap
            overlay

    Example outputs:
        gradcam_full_COVID.png
        gradcam_full_Lung_Opacity.png
        gradcam_full_Normal.png
        gradcam_full_Viral_Pneumonia.png
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths = []

    class_names = [
        "COVID",
        "Lung_Opacity",
        "Normal",
        "Viral Pneumonia",
    ]

    for class_name in class_names:

        class_frame = frame[
            frame[CLASS_COLUMN] == class_name
        ]

        if class_frame.empty:
            print(
                f"No examples available for "
                f"{class_name}"
            )
            continue

        n_samples = min(
            samples_per_class,
            len(class_frame),
        )

        sample = class_frame.sample(
            n=n_samples,
            random_state=random_state,
        )

        fig, axes = plt.subplots(
            n_samples,
            3,
            figsize=(10, 3.5 * n_samples),
            squeeze=False,
        )

        for row_idx, (_, row) in enumerate(
            sample.iterrows()
        ):
            image_path = str(
                row[IMAGE_PATH_COLUMN]
            )

            mask_path = str(
                row[MASK_PATH_COLUMN]
            )

            true_label = str(
                row[CLASS_COLUMN]
            )

            # --------------------------------------------------
            # Exact input seen by CNN
            # --------------------------------------------------
            model_input = load_cnn_gradcam_input(
                image_path=image_path,
                mask_path=mask_path,
                image_size=image_size,
                region=region,
                mask_threshold=mask_threshold,
            )

            # --------------------------------------------------
            # Lung mask in the SAME coordinate frame
            # --------------------------------------------------
            mask = read_grayscale(
                mask_path
            )

            if region == "lung_roi":
                original_image = read_grayscale(
                    image_path
                )

                mask = extract_lung_roi_mask(
                    mask=mask,
                    image_shape=np.squeeze(
                        original_image
                    ).shape,
                    target_size=image_size,
                    mask_threshold=mask_threshold,
                )

            else:
                mask = resize_mask(
                    mask,
                    image_size,
                )

            # --------------------------------------------------
            # Grad-CAM
            # --------------------------------------------------
            result = gradcam_for_image(
                model=model,
                image_array=model_input,
                mask=mask,
                target_layer_name=target_layer,
                mask_threshold=mask_threshold,
            )

            predicted_label = result[
                "predicted_label"
            ]

            lung_fraction = result[
                "lung_fraction"
            ]

            heatmap = result[
                "heatmap"
            ]

            overlay = result[
                "overlay"
            ]

            # --------------------------------------------------
            # Column 1: actual CNN input
            # --------------------------------------------------
            axes[row_idx, 0].imshow(
                np.squeeze(model_input),
                cmap="gray",
            )

            axes[row_idx, 0].set_title(
                f"Input ({region})\n"
                f"True: {true_label}"
            )

            # --------------------------------------------------
            # Column 2: Grad-CAM
            # --------------------------------------------------
            axes[row_idx, 1].imshow(
                heatmap,
                cmap="jet",
                vmin=0,
                vmax=1,
            )

            axes[row_idx, 1].set_title(
                "Grad-CAM heatmap"
            )

            # --------------------------------------------------
            # Column 3: overlay
            # --------------------------------------------------
            axes[row_idx, 2].imshow(
                overlay
            )

            # Draw lung contour
            axes[row_idx, 2].contour(
                mask > mask_threshold,
                levels=[0.5],
            )

            if (
                lung_fraction is not None
                and not np.isnan(lung_fraction)
            ):
                attention_text = (
                    f"{lung_fraction * 100:.2f}%"
                )
            else:
                attention_text = "N/A"

            axes[row_idx, 2].set_title(
                f"Pred: {predicted_label}\n"
                f"Lung attention: "
                f"{attention_text}"
            )

            for axis in axes[row_idx]:
                axis.axis("off")

        pretty_class_name = (
            class_name
            .replace(" ", "_")
        )

        output_path = (
            output_dir
            / (
                f"gradcam_{region}_"
                f"{pretty_class_name}.png"
            )
        )

        fig.suptitle(
            f"Grad-CAM — cnn_simple — "
            f"region={region} — "
            f"{class_name}",
            fontsize=14,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

        saved_paths.append(
            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    return saved_paths

if __name__ == "__main__":
    raise SystemExit(main())