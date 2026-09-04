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
        "--horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().horizontal_flip,
        help=(
            "Include random horizontal flipping in the augmentation pipeline "
            "(only applies when --augment is set). Use --no-horizontal-flip to "
            "test whether flipping hurts or helps, e.g. on asymmetric anatomy."
        ),
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        dest="use_class_weight",
        help=(
            "Weight the loss inversely to class frequency during training, to "
            "counter the dataset's class imbalance (Normal/Lung_Opacity outnumber "
            "COVID/Viral Pneumonia)."
        ),
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--mask-lungs",
        action="store_true",
        help="Zero out everything outside the lung mask before feeding images to the model.",
    )
    mask_group.add_argument(
        "--mask-only",
        action="store_true",
        help=(
            "Feed the segmentation mask itself (lung silhouette) instead of the X-ray "
            "image, discarding all pixel-intensity information. Tests how much class "
            "signal comes from lung shape/geometry alone."
        ),
    )
    parser.add_argument(
        "--mask-threshold", type=int, default=TransferConfig().mask_threshold
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        dest="unfreeze_backbone",
        help=(
            "Train the full EfficientNet from the start (single-phase). "
            "Mutually exclusive with --fine-tune, which keeps phase 1 frozen."
        ),
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help=(
            "After the frozen-head phase, unfreeze the backbone and continue "
            "training end-to-end at a lower learning rate."
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
        "--reduce-lr-on-plateau",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().reduce_lr_on_plateau,
        help=(
            "During fine-tuning, reduce the learning rate when val_loss stops "
            "improving (ignored when --fine-tune is not set)."
        ),
    )
    parser.add_argument(
        "--reduce-lr-factor", type=float, default=TransferConfig().reduce_lr_factor
    )
    parser.add_argument(
        "--reduce-lr-patience", type=int, default=TransferConfig().reduce_lr_patience
    )
    parser.add_argument(
        "--reduce-lr-min-lr", type=float, default=TransferConfig().reduce_lr_min_lr
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().pretrained,
        help="Load ImageNet weights (default) or start from random weights.",
    )
    parser.add_argument(
        "--track-train-f1",
        action="store_true",
        help=(
            "Also compute macro F1 on the training set after every epoch (in "
            "addition to validation), at the cost of an extra prediction pass "
            "per epoch. Useful to compare train vs val F1 curves for over/"
            "underfitting."
        ),
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=TransferConfig().save_checkpoints,
        help=(
            "Save a model checkpoint after every epoch under "
            "<models-dir>/checkpoints/<model-name>/, so training can be resumed."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint for --model-name, if one exists.",
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
    if args.fine_tune and args.unfreeze_backbone:
        build_parser().error("--fine-tune and --unfreeze-backbone are mutually exclusive")

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
            freeze_backbone=not args.unfreeze_backbone,
            pretrained=args.pretrained,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_learning_rate=args.fine_tune_learning_rate,
            fine_tune_early_stopping_patience=args.fine_tune_early_stopping_patience,
            reduce_lr_on_plateau=args.reduce_lr_on_plateau,
            reduce_lr_factor=args.reduce_lr_factor,
            reduce_lr_patience=args.reduce_lr_patience,
            reduce_lr_min_lr=args.reduce_lr_min_lr,
            augment=args.augment,
            horizontal_flip=args.horizontal_flip,
            use_class_weight=args.use_class_weight,
            mask_lungs=args.mask_lungs,
            mask_only=args.mask_only,
            mask_threshold=args.mask_threshold,
            track_train_f1=args.track_train_f1,
            save_checkpoints=args.save_checkpoints,
            random_state=args.seed,
        ),
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        save=not args.dry_run,
        resume=args.resume,
    )

    print(format_transfer_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
