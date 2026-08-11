from __future__ import annotations

from pathlib import Path

import pandas as pd
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.training import build_raw_manifest


def test_build_raw_manifest_lists_every_raw_image(manifest: pd.DataFrame) -> None:
    assert list(manifest.columns) == ["class", "image_path", "mask_path"]
    assert len(manifest) == IMAGES_PER_CLASS * len(CLASS_FOLDERS)
    assert set(manifest["class"]) == set(CLASS_FOLDERS)


def test_build_raw_manifest_excludes_redundant_files(raw_dir: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "redundant_images.csv"
    pd.DataFrame(
        {"class": ["COVID"], "redundant_file": ["COVID-0.png"]}
    ).to_csv(csv_path, index=False)

    manifest = build_raw_manifest(
        raw_dir=raw_dir, class_folders=CLASS_FOLDERS, redundant_csv=csv_path
    )

    assert len(manifest) == IMAGES_PER_CLASS * len(CLASS_FOLDERS) - 1
    assert not any(Path(path).name == "COVID-0.png" for path in manifest["image_path"])


def test_build_raw_manifest_without_redundant_csv_keeps_everything(
    raw_dir: Path,
) -> None:
    manifest = build_raw_manifest(raw_dir=raw_dir, class_folders=CLASS_FOLDERS)

    assert len(manifest) == IMAGES_PER_CLASS * len(CLASS_FOLDERS)
