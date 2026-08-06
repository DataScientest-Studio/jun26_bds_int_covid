from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import (
    ARRAY_DIR,
    CLASS_FOLDERS,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
)
from .config import PreprocessConfig, SplitConfig
from .pipeline import format_report, run_preprocessing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build model-ready arrays from the raw chest X-ray dataset."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--array-dir", type=Path, default=ARRAY_DIR)
    parser.add_argument("--redundant-csv", type=Path, default=None)
    parser.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, default=list(CLASS_NAMES)
    )
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--lung-mask", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--val-size", type=float, default=SplitConfig.val_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    result = run_preprocessing(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        array_dir=args.array_dir,
        class_folders={name: CLASS_FOLDERS[name] for name in args.classes},
        redundant_csv=args.redundant_csv,
        split_config=SplitConfig(
            val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        ),
        preprocess_config=PreprocessConfig(
            target_size=tuple(args.image_size),
            apply_clahe=args.clahe,
            apply_lung_mask=args.lung_mask,
            augment=args.augment,
            random_state=args.seed,
        ),
        copy_raw_files=not args.skip_copy,
        save=not args.dry_run,
    )

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
