from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR
from ..preprocessing.config import SplitConfig
from .config import TransferConfig
from .lung_only import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    LungOnlyPreprocessConfig,
    run_lung_only_training,
)
from .pipeline import format_transfer_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize background-free, rotation-balanced lung images and train "
            "an EfficientNetB0 classifier with early stopping and checkpoints."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR / "transfer_learning"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=SplitConfig.val_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze the backbone and continue training at a lower learning rate.",
    )
    parser.add_argument(
        "--fine-tune-from-best",
        action="store_true",
        help=(
            "Load the checkpoint with the highest validation macro F1 and start "
            "fine-tuning from there, skipping any further frozen-head training."
        ),
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=TransferConfig().fine_tune_epochs,
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=TransferConfig().fine_tune_learning_rate,
    )
    parser.add_argument(
        "--fine-tune-early-stopping-patience",
        type=int,
        default=TransferConfig().fine_tune_early_stopping_patience,
    )
    parser.add_argument(
        "--fine-tune-unfreeze-layers",
        type=int,
        default=TransferConfig().fine_tune_unfreeze_layers,
    )
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().reduce_lr_on_plateau,
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fine_tune_from_best and not args.fine_tune:
        build_parser().error("--fine-tune-from-best requires --fine-tune")
    image_size = tuple(args.image_size)
    prepared, trained = run_lung_only_training(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split_config=SplitConfig(
            val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        ),
        preprocess_config=LungOnlyPreprocessConfig(
            image_size=image_size,
            rotation_degrees=args.rotation_degrees,
            mask_threshold=args.mask_threshold,
            random_state=args.seed,
        ),
        training_config=TransferConfig(
            backbone="efficientnetb0",
            image_size=image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            pretrained=not args.no_pretrained,
            augment=False,
            horizontal_flip=False,
            balance_classes=False,
            use_class_weight=True,
            mask_lungs=False,
            early_stopping_patience=args.early_stopping_patience,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_learning_rate=args.fine_tune_learning_rate,
            fine_tune_early_stopping_patience=args.fine_tune_early_stopping_patience,
            fine_tune_unfreeze_layers=args.fine_tune_unfreeze_layers,
            reduce_lr_on_plateau=args.reduce_lr_on_plateau,
            save_checkpoints=True,
            random_state=args.seed,
        ),
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        prepare=not args.skip_prepare,
        save=not args.dry_run,
        resume=args.resume,
        fine_tune_from_best=args.fine_tune_from_best,
        verbose=1,
    )
    print(f"Processed images: {prepared.output_dir}")
    print(f"Processed manifest: {prepared.manifest_path}")
    print(format_transfer_report(trained))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
