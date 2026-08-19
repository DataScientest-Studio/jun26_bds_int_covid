from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tl_helpers import CLASS_FOLDERS, add_synthetic_masks, build_processed_dataset

from covid_xray.preprocessing import SplitConfig, Splits, build_manifest, split_manifest


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    return build_processed_dataset(tmp_path / "processed")


@pytest.fixture
def manifest(processed_dir: Path) -> pd.DataFrame:
    return build_manifest(processed_dir=processed_dir, class_folders=CLASS_FOLDERS)


@pytest.fixture
def manifest_with_masks(processed_dir: Path) -> pd.DataFrame:
    add_synthetic_masks(processed_dir)
    return build_manifest(processed_dir=processed_dir, class_folders=CLASS_FOLDERS)


@pytest.fixture
def splits(manifest: pd.DataFrame) -> Splits:
    return split_manifest(manifest, SplitConfig())
