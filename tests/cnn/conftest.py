from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from cnn_helpers import CLASS_FOLDERS, build_raw_dataset

from covid_xray.training import build_raw_manifest


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return build_raw_dataset(tmp_path / "raw")


@pytest.fixture
def manifest(raw_dir: Path) -> pd.DataFrame:
    return build_raw_manifest(raw_dir=raw_dir, class_folders=CLASS_FOLDERS)
