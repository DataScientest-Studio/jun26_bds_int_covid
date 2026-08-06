from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from covid_xray.preprocessing import (
    ArrayDataset,
    check_counts,
    compare_all,
    compare_with_saved,
    comparison_table,
    format_comparison_report,
    format_count_check,
    save_arrays,
)


def test_compare_with_saved_detects_identical_rebuild(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"
    save_arrays(small_datasets, directory)

    comparisons = compare_all(small_datasets, directory)

    assert all(comparison.matches for comparison in comparisons)
    assert "No data lost" in format_comparison_report(comparisons)


def test_compare_with_saved_reports_missing_previous_save(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    comparison = compare_with_saved("train", small_datasets["train"], tmp_path)

    assert not comparison.has_previous
    assert not comparison.matches
    assert comparison.previous_samples is None
    assert "No previous save on disk" in format_comparison_report([comparison])


def test_compare_with_saved_reports_pixel_changes(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"
    save_arrays(small_datasets, directory)
    changed = small_datasets["train"]
    changed.X = changed.X.copy()
    changed.X[0] = np.clip(changed.X[0] + 0.25, 0, 1)

    comparison = compare_with_saved("train", changed, directory)

    assert comparison.same_shape
    assert comparison.same_labels
    assert not comparison.same_values
    assert comparison.changed_samples == 1
    assert comparison.max_absolute_difference > 0
    assert "Max pixel diff" in format_comparison_report([comparison])


def test_compare_with_saved_detects_relabelled_samples(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"
    save_arrays(small_datasets, directory)
    relabelled = small_datasets["train"]
    relabelled.y = relabelled.y.copy()
    relabelled.y[0] = "Normal" if relabelled.y[0] != "Normal" else "COVID"

    comparison = compare_with_saved("train", relabelled, directory)

    assert comparison.same_shape
    assert not comparison.same_labels
    assert not comparison.matches


def test_compare_with_saved_reports_sample_count_change(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"
    save_arrays(small_datasets, directory)
    train = small_datasets["train"]
    shrunk = ArrayDataset(X=train.X[:-1], y=train.y[:-1])

    comparison = compare_with_saved("train", shrunk, directory)

    assert not comparison.same_shape
    assert comparison.previous_samples == shrunk.n_samples + 1
    assert "Sample count change" in format_comparison_report([comparison])


def test_comparison_table_has_one_row_per_split(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"
    save_arrays(small_datasets, directory)

    table = comparison_table(compare_all(small_datasets, directory))

    assert isinstance(table, pd.DataFrame)
    assert list(table["split"]) == ["train", "val", "test"]


def test_check_counts_flags_mismatch(manifest: pd.DataFrame) -> None:
    matching = check_counts(len(manifest), len(manifest), len(manifest))
    mismatched = check_counts(len(manifest), len(manifest) - 1, len(manifest) - 1)

    assert matching.manifest_matches_processed
    assert matching.fresh_matches_manifest
    assert "WARNING" not in format_count_check(matching)
    assert not mismatched.manifest_matches_processed
    assert "WARNING" in format_count_check(mismatched)
