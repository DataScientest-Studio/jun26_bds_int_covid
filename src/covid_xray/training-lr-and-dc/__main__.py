from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import CLASS_NAMES, MODELS_DIR, RANDOM_STATE, RAW_DIR
from ..preprocessing.config import SplitConfig
from .config import BaselineConfig, DEFAULT_BASELINE_IMAGE_SIZE
from .pipeline import BASELINE_REPORTS_DIR, format_baseline_report, run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a majority-class dummy baseline and a logistic regression "
            "baseline directly on raw chest X-ray images."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--redundant-csv", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=BASELINE_REPORTS_DIR)
    parser.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, default=list(CLASS_NAMES)
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=DEFAULT_BASELINE_IMAGE_SIZE)
    parser.add_argument("--val-size", type=float, default=SplitConfig.val_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ..config import CLASS_FOLDERS

    args = build_parser().parse_args(argv)

    result = run_baseline(
        raw_dir=args.raw_dir,
        class_folders={name: CLASS_FOLDERS[name] for name in args.classes},
        redundant_csv=args.redundant_csv,
        split_config=SplitConfig(
            val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        ),
        config=BaselineConfig(image_size=tuple(args.image_size), random_state=args.seed),
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        save=not args.dry_run,
    )

    print(format_baseline_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
