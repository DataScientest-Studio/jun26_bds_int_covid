from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple

import pandas as pd

from ..config import (
    CLASS_FOLDERS,
    IMAGE_GLOB,
    IMAGES_SUBDIR,
    MASKS_SUBDIR,
    PROCESSED_DIR,
    RAW_DIR,
)

REDUNDANT_CLASS_COLUMN = "class"
REDUNDANT_FILE_COLUMN = "redundant_file"


@dataclass(frozen=True)
class CopyResult:
    class_name: str
    copied: int
    skipped: int
    masks_copied: int
    masks_missing: int


def load_redundant_files(csv_path: Path | str) -> Set[Tuple[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Redundancy report not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = {REDUNDANT_CLASS_COLUMN, REDUNDANT_FILE_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    return set(
        zip(
            frame[REDUNDANT_CLASS_COLUMN].astype(str),
            frame[REDUNDANT_FILE_COLUMN].astype(str),
        )
    )


def copy_class(
    class_name: str,
    folder_name: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    redundant_files: Iterable[Tuple[str, str]] = (),
) -> CopyResult:
    source_dir = Path(raw_dir) / folder_name
    target_dir = Path(processed_dir) / folder_name
    source_images = source_dir / IMAGES_SUBDIR
    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing image folder: {source_images}")

    (target_dir / IMAGES_SUBDIR).mkdir(parents=True, exist_ok=True)
    (target_dir / MASKS_SUBDIR).mkdir(parents=True, exist_ok=True)

    excluded = set(redundant_files)
    copied = skipped = masks_copied = masks_missing = 0

    for image_path in sorted(source_images.glob(IMAGE_GLOB)):
        if (class_name, image_path.name) in excluded:
            skipped += 1
            continue

        shutil.copy2(image_path, target_dir / IMAGES_SUBDIR / image_path.name)
        mask_path = source_dir / MASKS_SUBDIR / image_path.name
        if mask_path.exists():
            shutil.copy2(mask_path, target_dir / MASKS_SUBDIR / image_path.name)
            masks_copied += 1
        else:
            masks_missing += 1
        copied += 1

    return CopyResult(
        class_name=class_name,
        copied=copied,
        skipped=skipped,
        masks_copied=masks_copied,
        masks_missing=masks_missing,
    )


def copy_dataset(
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    redundant_files: Iterable[Tuple[str, str]] = (),
) -> Dict[str, CopyResult]:
    excluded = set(redundant_files)
    return {
        class_name: copy_class(
            class_name=class_name,
            folder_name=folder_name,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            redundant_files=excluded,
        )
        for class_name, folder_name in class_folders.items()
    }


def count_files(
    root: Path | str,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    subdir: str = IMAGES_SUBDIR,
) -> Dict[str, int]:
    root = Path(root)
    return {
        class_name: len(list((root / folder_name / subdir).glob(IMAGE_GLOB)))
        for class_name, folder_name in class_folders.items()
    }


def count_images_and_masks(
    root: Path | str = PROCESSED_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
) -> Dict[str, int]:
    images = count_files(root, class_folders, IMAGES_SUBDIR)
    masks = count_files(root, class_folders, MASKS_SUBDIR)
    return {"images": sum(images.values()), "masks": sum(masks.values())}


def format_copy_report(results: Mapping[str, CopyResult]) -> str:
    lines = [
        f"{result.class_name}: copied {result.copied}, skipped {result.skipped}"
        + (f", masks missing {result.masks_missing}" if result.masks_missing else "")
        for result in results.values()
    ]
    total_copied = sum(result.copied for result in results.values())
    total_skipped = sum(result.skipped for result in results.values())
    lines.append(f"Total: copied {total_copied}, skipped {total_skipped}")
    return "\n".join(lines)
