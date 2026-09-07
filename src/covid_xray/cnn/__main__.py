from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import CLASS_NAMES, MODELS_DIR, RANDOM_STATE, RAW_DIR
from ..preprocessing.config import SplitConfig
from ..training.config import REGIONS
from .config import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    DEFAULT_LENET_VARIANT,
    DEFAULT_REGION,
    LENET_VARIANTS,
    LUNG_NORMALIZATIONS,
    DEFAULT_LUNG_NORMALIZATION,
    CNNConfig,
    default_image_size,
)
from .pipeline import CNN_REPORTS_DIR, format_cnn_report, run_cnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small CNN from random initialization on chest X-rays. "
            "Use --region to restrict the network to the lung field or to the "
            "background only. The background run is the control: a full-image "
            "score that does not clearly exceed it is not evidence the network "
            "learned anything about lungs."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--redundant-csv", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=CNN_REPORTS_DIR)
    parser.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, default=list(CLASS_NAMES)
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default=DEFAULT_REGION,
        help="Which pixels the network may see (default: full).",
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
        help=(
            "simple = three-layer CNN baseline (16, 32, 64 filters); "
            "scratch = VGG-style stack with a global-average-pooling head; "
            "lenet = LeNet-5, a low-capacity floor (default: scratch)."
        ),
    )
    parser.add_argument(
        "--lenet-variant",
        choices=LENET_VARIANTS,
        default=DEFAULT_LENET_VARIANT,
        help=(
            "original = tanh + average pooling, as published; "
            "modern = ReLU + max pooling. Ignored unless --architecture lenet."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        help=(
            "Defaults to 128x128 for scratch and 32x32 for lenet, which pools "
            "only twice and blows up its dense head at higher resolutions."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=CNNConfig().batch_size)
    parser.add_argument("--epochs", type=int, default=CNNConfig().epochs)
    parser.add_argument("--learning-rate", type=float, default=CNNConfig().learning_rate)
    parser.add_argument("--dropout-rate", type=float, default=CNNConfig().dropout_rate)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--class-weight",
        action=argparse.BooleanOptionalAction,
        default=CNNConfig().class_weight,
        help="Weight the loss by inverse class frequency (default: on).",
    )
    parser.add_argument("--val-size", type=float, default=SplitConfig.val_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Base name for the saved model and report files; --region is "
            "appended. Defaults to the architecture name."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--lung-normalization",
        choices=LUNG_NORMALIZATIONS,
        default=DEFAULT_LUNG_NORMALIZATION,
        help=(
            "Optional lungs-only normalization: "
            "none, geometry, intensity, or both."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ..config import CLASS_FOLDERS

    args = build_parser().parse_args(argv)

    # Resolve the resolution default only after the architecture is known.
    image_size = (
        tuple(args.image_size)
        if args.image_size is not None
        else default_image_size(args.architecture)
    )

    result = run_cnn(
        raw_dir=args.raw_dir,
        class_folders={name: CLASS_FOLDERS[name] for name in args.classes},
        redundant_csv=args.redundant_csv,
        split_config=SplitConfig(
            val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        ),
        config=CNNConfig(
            image_size=image_size,
            architecture=args.architecture,
            lenet_variant=args.lenet_variant,
            region=args.region,
            lung_normalization=args.lung_normalization,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            augment=args.augment,
            class_weight=args.class_weight,
            random_state=args.seed,
        ),
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        save=not args.dry_run,
    )

    print(format_cnn_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
