from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import CLASS_NAMES, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from ..preprocessing.config import SplitConfig
from .config import BACKBONES, DEFAULT_BACKBONE, TransferConfig
from .pipeline import (
    DEFAULT_MODEL_NAME,
    TRANSFER_REPORTS_DIR,
    format_transfer_report,
    run_transfer_learning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an EfficientNet transfer-learning classifier on chest X-rays. "
            "By default the ImageNet-pretrained backbone stays frozen and only a "
            "small classification head is trained on top of it."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=TRANSFER_REPORTS_DIR)
    parser.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, default=list(CLASS_NAMES)
    )
    parser.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    parser.add_argument("--image-size", type=int, nargs=2, default=TransferConfig().image_size)
    parser.add_argument("--batch-size", type=int, default=TransferConfig().batch_size)
    parser.add_argument("--epochs", type=int, default=TransferConfig().epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=TransferConfig().learning_rate
    )
    parser.add_argument("--dense-units", type=int, default=TransferConfig().dense_units)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().pretrained,
        help="Load ImageNet weights (default) or start from random weights.",
    )
    parser.add_argument("--val-size", type=float, default=SplitConfig.val_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Name used for the saved model file and report filenames.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ..config import CLASS_FOLDERS

    args = build_parser().parse_args(argv)

    result = run_transfer_learning(
        processed_dir=args.processed_dir,
        class_folders={name: CLASS_FOLDERS[name] for name in args.classes},
        split_config=SplitConfig(
            val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        ),
        config=TransferConfig(
            backbone=args.backbone,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dense_units=args.dense_units,
            pretrained=args.pretrained,
            augment=args.augment,
            random_state=args.seed,
        ),
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        save=not args.dry_run,
    )

    print(format_transfer_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
