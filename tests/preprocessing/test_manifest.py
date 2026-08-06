from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import (
    SplitConfig,
    build_manifest,
    class_proportions,
    missing_paths,
    split_manifest,
    split_summary,
)


def test_build_manifest_lists_every_image_with_its_mask(
    manifest: pd.DataFrame,
) -> None:
    assert list(manifest.columns) == ["class", "image_path", "mask_path"]
    assert len(manifest) == IMAGES_PER_CLASS * len(CLASS_FOLDERS)
    assert set(manifest["class"]) == set(CLASS_FOLDERS)
    assert missing_paths(manifest) == {"image_path": 0, "mask_path": 0}
    assert all(
        Path(row.image_path).name == Path(row.mask_path).name
        for row in manifest.itertuples()
    )


def test_build_manifest_is_deterministic(processed_dir: Path) -> None:
    first = build_manifest(processed_dir, CLASS_FOLDERS)
    second = build_manifest(processed_dir, CLASS_FOLDERS)

    pd.testing.assert_frame_equal(first, second)


def test_build_manifest_on_empty_folder_returns_empty_frame(tmp_path: Path) -> None:
    result = build_manifest(processed_dir=tmp_path, class_folders=CLASS_FOLDERS)

    assert result.empty
    assert list(result.columns) == ["class", "image_path", "mask_path"]


def test_missing_paths_counts_absent_files(manifest: pd.DataFrame) -> None:
    Path(manifest.loc[0, "image_path"]).unlink()

    assert missing_paths(manifest) == {"image_path": 1, "mask_path": 0}


def test_split_manifest_preserves_all_rows_without_overlap(
    manifest: pd.DataFrame,
) -> None:
    splits = split_manifest(manifest, SplitConfig())

    assert splits.total == len(manifest)
    train_paths = set(splits.train["image_path"])
    val_paths = set(splits.val["image_path"])
    test_paths = set(splits.test["image_path"])
    assert not train_paths & val_paths
    assert not train_paths & test_paths
    assert not val_paths & test_paths
    assert train_paths | val_paths | test_paths == set(manifest["image_path"])


def test_split_manifest_respects_requested_sizes(manifest: pd.DataFrame) -> None:
    splits = split_manifest(manifest, SplitConfig(val_size=0.2, test_size=0.2))

    assert len(splits.val) == pytest.approx(0.2 * len(manifest), abs=1)
    assert len(splits.test) == pytest.approx(0.2 * len(manifest), abs=1)
    assert len(splits.train) == pytest.approx(0.6 * len(manifest), abs=1)


def test_split_manifest_keeps_class_proportions(manifest: pd.DataFrame) -> None:
    splits = split_manifest(manifest, SplitConfig())
    reference = class_proportions(manifest)

    for _, frame in splits.items():
        pd.testing.assert_series_equal(
            class_proportions(frame), reference, atol=0.05, check_names=False
        )


def test_split_summary_covers_every_split(manifest: pd.DataFrame) -> None:
    summary = split_summary(split_manifest(manifest, SplitConfig()))

    assert list(summary.columns) == ["train", "val", "test"]
    assert set(summary.index) == set(CLASS_FOLDERS)


def test_split_manifest_is_reproducible(manifest: pd.DataFrame) -> None:
    first = split_manifest(manifest, SplitConfig(random_state=7))
    second = split_manifest(manifest, SplitConfig(random_state=7))
    other = split_manifest(manifest, SplitConfig(random_state=8))

    assert list(first.train["image_path"]) == list(second.train["image_path"])
    assert list(first.train["image_path"]) != list(other.train["image_path"])


def test_splits_support_lookup_by_name(manifest: pd.DataFrame) -> None:
    splits = split_manifest(manifest, SplitConfig())

    assert splits["train"] is splits.train
    with pytest.raises(KeyError):
        splits["holdout"]


def test_split_manifest_rejects_empty_manifest() -> None:
    empty = pd.DataFrame(columns=["class", "image_path", "mask_path"])

    with pytest.raises(ValueError):
        split_manifest(empty)


def test_split_config_rejects_impossible_ratios() -> None:
    with pytest.raises(ValueError):
        SplitConfig(val_size=0.6, test_size=0.5)

    with pytest.raises(ValueError):
        SplitConfig(val_size=0.0)
