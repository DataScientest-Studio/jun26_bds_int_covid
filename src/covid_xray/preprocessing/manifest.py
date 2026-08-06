from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import (
    CLASS_COLUMN,
    CLASS_FOLDERS,
    IMAGE_GLOB,
    IMAGE_PATH_COLUMN,
    IMAGES_SUBDIR,
    MANIFEST_COLUMNS,
    MASK_PATH_COLUMN,
    MASKS_SUBDIR,
    PROCESSED_DIR,
)
from .config import SplitConfig


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __getitem__(self, split: str) -> pd.DataFrame:
        try:
            return getattr(self, split)
        except AttributeError as error:
            raise KeyError(split) from error

    def items(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        return iter(
            (("train", self.train), ("val", self.val), ("test", self.test))
        )

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def build_manifest(
    processed_dir: Path | str = PROCESSED_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    records = []
    for class_name, folder_name in class_folders.items():
        image_dir = processed_dir / folder_name / IMAGES_SUBDIR
        mask_dir = processed_dir / folder_name / MASKS_SUBDIR
        for image_path in sorted(image_dir.glob(IMAGE_GLOB)):
            records.append(
                {
                    CLASS_COLUMN: class_name,
                    IMAGE_PATH_COLUMN: str(image_path),
                    MASK_PATH_COLUMN: str(mask_dir / image_path.name),
                }
            )

    return pd.DataFrame(records, columns=list(MANIFEST_COLUMNS))


def split_manifest(
    manifest: pd.DataFrame, config: SplitConfig = SplitConfig()
) -> Splits:
    if manifest.empty:
        raise ValueError("Cannot split an empty manifest")

    stratify_column = config.stratify_column
    train_df, holdout_df = train_test_split(
        manifest,
        test_size=config.holdout_size,
        stratify=manifest[stratify_column],
        random_state=config.random_state,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=config.test_size_within_holdout,
        stratify=holdout_df[stratify_column],
        random_state=config.random_state,
    )

    return Splits(train=train_df, val=val_df, test=test_df)


def class_proportions(
    frame: pd.DataFrame, class_column: str = CLASS_COLUMN
) -> pd.Series:
    return frame[class_column].value_counts(normalize=True).sort_index()


def split_summary(splits: Splits, class_column: str = CLASS_COLUMN) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: class_proportions(frame, class_column)
            for name, frame in splits.items()
        }
    )


def missing_paths(manifest: pd.DataFrame) -> Dict[str, int]:
    return {
        IMAGE_PATH_COLUMN: int(
            sum(not Path(path).exists() for path in manifest[IMAGE_PATH_COLUMN])
        ),
        MASK_PATH_COLUMN: int(
            sum(not Path(path).exists() for path in manifest[MASK_PATH_COLUMN])
        ),
    }
