from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import (
    ArrayDataset,
    PreprocessConfig,
    SplitConfig,
    array_paths,
    build_array_dataset,
    build_split_arrays,
    encode_labels,
    format_failures,
    load_arrays,
    save_arrays,
    split_manifest,
)


def test_build_array_dataset_stacks_images_and_labels(manifest: pd.DataFrame) -> None:
    dataset = build_array_dataset(manifest, PreprocessConfig(target_size=(32, 32)))

    assert dataset.X.shape == (len(manifest), 32, 32)
    assert dataset.y.shape == (len(manifest),)
    assert dataset.failed == []
    assert dataset.class_counts() == {name: IMAGES_PER_CLASS for name in CLASS_FOLDERS}
    assert "No files failed" in format_failures(dataset)


def test_build_array_dataset_collects_failures(manifest: pd.DataFrame) -> None:
    broken = manifest.copy()
    broken.loc[broken.index[0], "image_path"] = "/does/not/exist.png"

    dataset = build_array_dataset(broken, PreprocessConfig(target_size=(32, 32)))

    assert dataset.n_samples == len(manifest) - 1
    assert len(dataset.failed) == 1
    assert dataset.failed[0][0] == "/does/not/exist.png"
    assert "1 files failed to process" in format_failures(dataset)


def test_build_array_dataset_on_empty_frame_returns_empty_arrays() -> None:
    empty = pd.DataFrame(columns=["class", "image_path", "mask_path"])

    dataset = build_array_dataset(empty, PreprocessConfig(target_size=(32, 32)))

    assert dataset.X.shape == (0, 32, 32)
    assert dataset.n_samples == 0


def test_build_array_dataset_augmentation_is_seeded(manifest: pd.DataFrame) -> None:
    config = PreprocessConfig(target_size=(32, 32), augment=True, random_state=11)

    first = build_array_dataset(manifest, config)
    second = build_array_dataset(manifest, config)
    plain = build_array_dataset(manifest, PreprocessConfig(target_size=(32, 32)))

    assert np.array_equal(first.X, second.X)
    assert not np.allclose(first.X, plain.X)


def test_build_split_arrays_augments_only_selected_splits(
    manifest: pd.DataFrame,
) -> None:
    splits = split_manifest(manifest, SplitConfig())
    config = PreprocessConfig(target_size=(32, 32), augment=True)

    datasets = build_split_arrays(splits, config, augment_splits=("train",))
    reference = {
        name: build_array_dataset(frame, PreprocessConfig(target_size=(32, 32)))
        for name, frame in splits.items()
    }

    assert set(datasets) == {"train", "val", "test"}
    assert np.allclose(datasets["val"].X, reference["val"].X)
    assert np.allclose(datasets["test"].X, reference["test"].X)
    assert not np.allclose(datasets["train"].X, reference["train"].X)


def test_build_split_arrays_without_augmentation_matches_plain_build(
    manifest: pd.DataFrame,
) -> None:
    splits = split_manifest(manifest, SplitConfig())
    config = PreprocessConfig(target_size=(16, 16))

    datasets = build_split_arrays(splits, config)

    for name, frame in splits.items():
        assert np.allclose(
            datasets[name].X, build_array_dataset(frame, config).X
        )


def test_encode_labels_maps_known_classes() -> None:
    labels = np.array(["COVID", "Normal", "COVID"])

    assert list(encode_labels(labels)) == [0, 2, 0]


def test_encode_labels_rejects_unknown_classes() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        encode_labels(np.array(["Unknown"]))


def test_array_paths_follow_split_naming(tmp_path: Path) -> None:
    x_path, y_path = array_paths("train", tmp_path)

    assert x_path == tmp_path / "X_train.npy"
    assert y_path == tmp_path / "y_train.npy"


def test_save_and_load_arrays_round_trip(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "arrays"

    saved = save_arrays(small_datasets, directory)

    assert set(saved) == {"train", "val", "test"}
    for split, dataset in small_datasets.items():
        X, y = load_arrays(split, directory)
        assert np.array_equal(X, dataset.X)
        assert np.array_equal(y, dataset.y)


def test_save_arrays_creates_missing_directory(
    tmp_path: Path, small_datasets: Dict[str, ArrayDataset]
) -> None:
    directory = tmp_path / "nested" / "arrays"

    save_arrays(small_datasets, directory)

    assert directory.is_dir()


def test_load_arrays_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_arrays("train", tmp_path)
