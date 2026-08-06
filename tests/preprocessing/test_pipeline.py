from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import (
    METADATA_FILENAME,
    PreprocessConfig,
    SplitConfig,
    format_report,
    load_arrays,
    load_metadata,
    package_versions,
    run_preprocessing,
)
from covid_xray.preprocessing.__main__ import main

SMALL = PreprocessConfig(target_size=(16, 16))


def run_step(raw_dir: Path, tmp_path: Path, **overrides):
    defaults = dict(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        array_dir=tmp_path / "arrays",
        class_folders=CLASS_FOLDERS,
        preprocess_config=SMALL,
    )
    return run_preprocessing(**{**defaults, **overrides})


def test_run_preprocessing_produces_saved_splits(raw_dir: Path, tmp_path: Path) -> None:
    result = run_step(raw_dir, tmp_path)

    expected_total = IMAGES_PER_CLASS * len(CLASS_FOLDERS)
    assert result.total_samples == expected_total
    assert len(result.manifest) == expected_total
    assert set(result.saved_paths) == {"train", "val", "test"}
    assert result.failed == []
    assert result.counts.manifest_matches_processed
    assert result.counts.fresh_matches_manifest

    for split, dataset in result.datasets.items():
        X, y = load_arrays(split, tmp_path / "arrays")
        assert np.array_equal(X, dataset.X)
        assert np.array_equal(y, dataset.y)


def test_run_preprocessing_excludes_redundant_files(
    raw_dir: Path, tmp_path: Path
) -> None:
    csv_path = tmp_path / "redundant_images.csv"
    pd.DataFrame(
        {"class": ["COVID"], "redundant_file": ["COVID-0.png"]}
    ).to_csv(csv_path, index=False)

    result = run_step(raw_dir, tmp_path, redundant_csv=csv_path)

    assert result.copy_results["COVID"].skipped == 1
    assert result.total_samples == IMAGES_PER_CLASS * len(CLASS_FOLDERS) - 1
    assert not any(
        Path(path).name == "COVID-0.png" for path in result.manifest["image_path"]
    )


def test_run_preprocessing_dry_run_writes_nothing(
    raw_dir: Path, tmp_path: Path
) -> None:
    result = run_step(raw_dir, tmp_path, save=False)

    assert result.saved_paths == {}
    assert not (tmp_path / "arrays").exists()
    assert result.total_samples > 0


def test_run_preprocessing_can_skip_copying(raw_dir: Path, tmp_path: Path) -> None:
    run_step(raw_dir, tmp_path)

    rerun = run_step(raw_dir, tmp_path, copy_raw_files=False)

    assert rerun.copy_results == {}
    assert rerun.total_samples == IMAGES_PER_CLASS * len(CLASS_FOLDERS)


def test_rerun_matches_previous_save(raw_dir: Path, tmp_path: Path) -> None:
    run_step(raw_dir, tmp_path)

    rerun = run_step(raw_dir, tmp_path, copy_raw_files=False)

    assert all(comparison.matches for comparison in rerun.comparisons)
    assert "No data lost" in format_report(rerun)


def test_changed_settings_are_reported_as_differences(
    raw_dir: Path, tmp_path: Path
) -> None:
    run_step(raw_dir, tmp_path)

    rerun = run_step(
        raw_dir,
        tmp_path,
        copy_raw_files=False,
        preprocess_config=PreprocessConfig(target_size=(16, 16), apply_clahe=True),
        save=False,
    )

    assert not all(comparison.matches for comparison in rerun.comparisons)
    assert "Differences detected" in format_report(rerun)


def test_metadata_records_settings_and_counts(raw_dir: Path, tmp_path: Path) -> None:
    result = run_step(
        raw_dir,
        tmp_path,
        split_config=SplitConfig(val_size=0.2, test_size=0.2, random_state=7),
    )

    saved = load_metadata(tmp_path / "arrays")

    assert saved == result.metadata
    assert saved["split_config"]["random_state"] == 7
    assert saved["preprocess_config"]["target_size"] == [16, 16]
    assert saved["samples"] == {
        split: dataset.n_samples for split, dataset in result.datasets.items()
    }
    assert set(saved["versions"]) >= {"python", "numpy", "pandas", "opencv-python"}
    assert json.loads((tmp_path / "arrays" / METADATA_FILENAME).read_text()) == saved


def test_load_metadata_requires_a_previous_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_package_versions_lists_core_dependencies() -> None:
    versions = package_versions()

    assert versions["numpy"] == np.__version__
    assert versions["pandas"] == pd.__version__


def test_format_report_covers_each_stage(raw_dir: Path, tmp_path: Path) -> None:
    report = format_report(run_step(raw_dir, tmp_path))

    assert "Total: copied" in report
    assert "Processed folder images" in report
    assert "TRAIN" in report
    assert "Saved arrays for" in report


def cli_args(raw_dir: Path, tmp_path: Path) -> list:
    return [
        "--raw-dir",
        str(raw_dir),
        "--processed-dir",
        str(tmp_path / "processed"),
        "--array-dir",
        str(tmp_path / "arrays"),
        "--classes",
        *CLASS_FOLDERS,
        "--image-size",
        "16",
        "16",
    ]


def test_cli_runs_the_step_end_to_end(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(cli_args(raw_dir, tmp_path) + ["--seed", "7"])

    assert exit_code == 0
    assert "Saved arrays for" in capsys.readouterr().out
    assert load_metadata(tmp_path / "arrays")["split_config"]["random_state"] == 7


def test_cli_dry_run_leaves_arrays_untouched(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(cli_args(raw_dir, tmp_path) + ["--dry-run"])

    capsys.readouterr()
    assert exit_code == 0
    assert not (tmp_path / "arrays").exists()
