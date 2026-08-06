from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    ARRAY_DIR,
    CLASS_COLUMN,
    IMAGE_PATH_COLUMN,
    LABEL_TO_ID,
    MASK_PATH_COLUMN,
)
from .augmentation import AugmentConfig, make_rng
from .config import PreprocessConfig
from .manifest import Splits
from .transforms import preprocess_xray


@dataclass
class ArrayDataset:
    X: np.ndarray
    y: np.ndarray
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def class_counts(self) -> Dict[str, int]:
        return pd.Series(self.y).value_counts().to_dict()


def array_paths(split: str, directory: Path | str = ARRAY_DIR) -> Tuple[Path, Path]:
    directory = Path(directory)
    return directory / f"X_{split}.npy", directory / f"y_{split}.npy"


def encode_labels(
    labels: np.ndarray, label_to_id: Mapping[str, int] = LABEL_TO_ID
) -> np.ndarray:
    unknown = sorted(set(labels) - set(label_to_id))
    if unknown:
        raise ValueError(f"Labels missing from mapping: {unknown}")
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def build_array_dataset(
    frame: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
    augment_config: AugmentConfig = AugmentConfig(),
    rng: Optional[random.Random] = None,
) -> ArrayDataset:
    if config.augment and rng is None:
        rng = make_rng(config.random_state)

    images: List[np.ndarray] = []
    labels: List[str] = []
    failed: List[Tuple[str, str]] = []

    for _, row in frame.iterrows():
        try:
            images.append(
                preprocess_xray(
                    row[IMAGE_PATH_COLUMN],
                    row[MASK_PATH_COLUMN],
                    config=config,
                    augment_config=augment_config,
                    rng=rng,
                )
            )
            labels.append(row[CLASS_COLUMN])
        except Exception as error:
            failed.append((str(row[IMAGE_PATH_COLUMN]), str(error)))

    if images:
        X = np.stack(images)
    else:
        X = np.empty((0, *config.target_size), dtype=np.float32)

    return ArrayDataset(X=X, y=np.array(labels), failed=failed)


def build_split_arrays(
    splits: Splits,
    config: PreprocessConfig = PreprocessConfig(),
    augment_config: AugmentConfig = AugmentConfig(),
    augment_splits: Tuple[str, ...] = ("train",),
) -> Dict[str, ArrayDataset]:
    datasets: Dict[str, ArrayDataset] = {}
    for name, frame in splits.items():
        split_config = replace(
            config, augment=config.augment and name in augment_splits
        )
        datasets[name] = build_array_dataset(
            frame,
            config=split_config,
            augment_config=augment_config,
            rng=make_rng(config.random_state) if split_config.augment else None,
        )
    return datasets


def save_arrays(
    datasets: Mapping[str, ArrayDataset], directory: Path | str = ARRAY_DIR
) -> Dict[str, Tuple[Path, Path]]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Tuple[Path, Path]] = {}
    for split, dataset in datasets.items():
        x_path, y_path = array_paths(split, directory)
        np.save(x_path, dataset.X)
        np.save(y_path, dataset.y)
        saved[split] = (x_path, y_path)
    return saved


def load_arrays(
    split: str, directory: Path | str = ARRAY_DIR
) -> Tuple[np.ndarray, np.ndarray]:
    x_path, y_path = array_paths(split, directory)
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError(f"Missing saved arrays for split '{split}' in {directory}")
    return np.load(x_path), np.load(y_path, allow_pickle=False)


def format_failures(dataset: ArrayDataset, limit: int = 10) -> str:
    if not dataset.failed:
        return "No files failed to process."

    lines = [f"{len(dataset.failed)} files failed to process:"]
    lines.extend(f"  {path}: {error}" for path, error in dataset.failed[:limit])
    if len(dataset.failed) > limit:
        lines.append(f"  ... and {len(dataset.failed) - limit} more")
    return "\n".join(lines)
