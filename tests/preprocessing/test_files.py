from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import (
    copy_dataset,
    count_images_and_masks,
    format_copy_report,
    load_redundant_files,
)


def test_load_redundant_files_returns_class_file_pairs(tmp_path: Path) -> None:
    csv_path = tmp_path / "redundant_images.csv"
    pd.DataFrame(
        {
            "class": ["COVID", "Normal"],
            "redundant_file": ["COVID-0.png", "Normal-1.png"],
        }
    ).to_csv(csv_path, index=False)

    assert load_redundant_files(csv_path) == {
        ("COVID", "COVID-0.png"),
        ("Normal", "Normal-1.png"),
    }


def test_load_redundant_files_rejects_unexpected_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"class": ["COVID"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="redundant_file"):
        load_redundant_files(csv_path)


def test_load_redundant_files_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_redundant_files(tmp_path / "missing.csv")


def test_copy_dataset_copies_images_and_masks(tmp_path: Path, raw_dir: Path) -> None:
    target = tmp_path / "processed"

    results = copy_dataset(
        raw_dir=raw_dir, processed_dir=target, class_folders=CLASS_FOLDERS
    )

    assert {name: result.copied for name, result in results.items()} == {
        name: IMAGES_PER_CLASS for name in CLASS_FOLDERS
    }
    assert count_images_and_masks(target, CLASS_FOLDERS) == {
        "images": IMAGES_PER_CLASS * len(CLASS_FOLDERS),
        "masks": IMAGES_PER_CLASS * len(CLASS_FOLDERS),
    }


def test_copy_dataset_skips_redundant_files(tmp_path: Path, raw_dir: Path) -> None:
    target = tmp_path / "processed"

    results = copy_dataset(
        raw_dir=raw_dir,
        processed_dir=target,
        class_folders=CLASS_FOLDERS,
        redundant_files={("COVID", "COVID-0.png"), ("COVID", "COVID-1.png")},
    )

    assert results["COVID"].skipped == 2
    assert results["COVID"].copied == IMAGES_PER_CLASS - 2
    assert results["Normal"].skipped == 0
    assert not (target / "COVID" / "images" / "COVID-0.png").exists()
    assert not (target / "COVID" / "masks" / "COVID-0.png").exists()


def test_copy_dataset_reports_missing_masks(tmp_path: Path, raw_dir: Path) -> None:
    (raw_dir / "Normal" / "masks" / "Normal-0.png").unlink()

    results = copy_dataset(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        class_folders=CLASS_FOLDERS,
    )

    assert results["Normal"].masks_missing == 1
    assert results["Normal"].copied == IMAGES_PER_CLASS


def test_copy_dataset_raises_for_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        copy_dataset(
            raw_dir=tmp_path / "missing",
            processed_dir=tmp_path / "processed",
            class_folders=CLASS_FOLDERS,
        )


def test_format_copy_report_totals_every_class(tmp_path: Path, raw_dir: Path) -> None:
    results = copy_dataset(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        class_folders=CLASS_FOLDERS,
        redundant_files={("COVID", "COVID-0.png")},
    )

    report = format_copy_report(results)

    assert "COVID: copied 19, skipped 1" in report
    assert f"Total: copied {IMAGES_PER_CLASS * len(CLASS_FOLDERS) - 1}" in report
