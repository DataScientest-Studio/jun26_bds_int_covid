from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tensorflow import keras

from ..config import CLASS_FOLDERS, PROCESSED_DIR, RANDOM_STATE
from .config import TransferConfig
from .gradcam import (
    DEFAULT_BACKBONE_LAYER,
    aggregate_lung_focus,
    aggregate_lung_focus_by_correctness,
    save_correctness_lung_focus_report,
    save_gradcam_grid,
    save_lung_focus_report,
    summarize_lung_focus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for a saved transfer-learning model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=TransferConfig().image_size)
    parser.add_argument("--backbone-layer", default=DEFAULT_BACKBONE_LAYER)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--lung-focus-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute the share of Grad-CAM attention that falls inside the lung mask.",
    )
    parser.add_argument(
        "--lung-focus-sample-size",
        type=int,
        default=200,
        help="Number of images to sample per run when computing the lung-focus report.",
    )
    parser.add_argument(
        "--apply-mask-to-input",
        action="store_true",
        help=(
            "Zero out the background before feeding images to the model, matching a "
            "model trained with --mask-lungs. Without this flag Grad-CAM sees the "
            "full, unmasked image."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ..preprocessing.manifest import build_manifest

    args = build_parser().parse_args(argv)

    model = keras.models.load_model(args.model_path)
    manifest = build_manifest(processed_dir=args.processed_dir, class_folders=CLASS_FOLDERS)
    image_size = tuple(args.image_size)

    output_path = save_gradcam_grid(
        model,
        manifest,
        args.output,
        image_size=image_size,
        backbone_layer_name=args.backbone_layer,
        samples_per_class=args.samples_per_class,
        random_state=args.seed,
        apply_mask_to_input=args.apply_mask_to_input,
    )
    print(f"Saved Grad-CAM grid to {output_path}")

    if args.lung_focus_report:
        summary = summarize_lung_focus(
            model,
            manifest,
            image_size=image_size,
            backbone_layer_name=args.backbone_layer,
            sample_size=args.lung_focus_sample_size,
            random_state=args.seed,
            apply_mask_to_input=args.apply_mask_to_input,
        )
        csv_path, png_path = save_lung_focus_report(
            summary, args.output.parent, args.model_path.stem
        )
        print(f"Saved lung-focus report to {csv_path} and {png_path}")
        print(aggregate_lung_focus(summary).round(3).to_string())

        correctness_csv_path, correctness_png_path = save_correctness_lung_focus_report(
            summary, args.output.parent, args.model_path.stem
        )
        print(f"Saved correctness lung-focus report to {correctness_csv_path} and {correctness_png_path}")
        print(aggregate_lung_focus_by_correctness(summary).round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
