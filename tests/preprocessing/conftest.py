from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, build_raw_dataset

from covid_xray.preprocessing import (
    ArrayDataset,
    PreprocessConfig,
    SplitConfig,
    build_array_dataset,
    build_manifest,
    copy_dataset,
    split_manifest,
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return build_raw_dataset(tmp_path / "raw")


@pytest.fixture
def processed_dir(tmp_path: Path, raw_dir: Path) -> Path:
    root = tmp_path / "processed"
    copy_dataset(raw_dir=raw_dir, processed_dir=root, class_folders=CLASS_FOLDERS)
    return root


@pytest.fixture
def manifest(processed_dir: Path) -> pd.DataFrame:
    return build_manifest(processed_dir=processed_dir, class_folders=CLASS_FOLDERS)


@pytest.fixture
def image_and_mask(raw_dir: Path) -> Tuple[Path, Path]:
    return (
        raw_dir / "COVID" / "images" / "COVID-0.png",
        raw_dir / "COVID" / "masks" / "COVID-0.png",
    )


@pytest.fixture
def small_datasets(manifest: pd.DataFrame) -> Dict[str, ArrayDataset]:
    splits = split_manifest(manifest, SplitConfig())
    return {
        name: build_array_dataset(frame, PreprocessConfig(target_size=(16, 16)))
        for name, frame in splits.items()
    }
