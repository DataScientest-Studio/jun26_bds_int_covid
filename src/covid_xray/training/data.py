from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from ..config import CLASS_COLUMN, CLASS_FOLDERS, IMAGE_PATH_COLUMN, RAW_DIR
from ..preprocessing.files import load_redundant_files
from ..preprocessing.manifest import build_manifest


def build_raw_manifest(
    raw_dir: Path | str = RAW_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    redundant_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    manifest = build_manifest(raw_dir, class_folders)
    if redundant_csv is None:
        return manifest

    redundant_files = load_redundant_files(redundant_csv)
    keys = list(
        zip(manifest[CLASS_COLUMN], manifest[IMAGE_PATH_COLUMN].map(lambda p: Path(p).name))
    )
    keep_mask = [key not in redundant_files for key in keys]
    return manifest[keep_mask].reset_index(drop=True)
