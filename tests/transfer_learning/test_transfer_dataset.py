from __future__ import annotations

import numpy as np
import pandas as pd

from covid_xray.config import LABEL_TO_ID
from covid_xray.preprocessing import Splits
from covid_xray.transfer_learning import TransferConfig
from covid_xray.transfer_learning.dataset import build_dataset, build_datasets

SMALL = TransferConfig(image_size=(32, 32), batch_size=4)


def test_build_dataset_yields_batches_with_expected_shape_and_range(
    manifest: pd.DataFrame,
) -> None:
    dataset = build_dataset(manifest, SMALL, shuffle=False, augment=False)
    images, labels = next(iter(dataset))

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert images.dtype.name == "float32"
    assert float(np.min(images.numpy())) >= 0.0
    assert float(np.max(images.numpy())) <= 255.0
    assert labels.shape[0] == images.shape[0]


def test_build_dataset_labels_match_manifest_encoding(manifest: pd.DataFrame) -> None:
    dataset = build_dataset(manifest, SMALL, shuffle=False, augment=False)
    all_labels = np.concatenate([labels.numpy() for _, labels in dataset])
    expected = manifest["class"].map(LABEL_TO_ID).to_numpy()

    assert np.array_equal(all_labels, expected)


def test_build_dataset_with_augmentation_keeps_shape(manifest: pd.DataFrame) -> None:
    dataset = build_dataset(manifest, SMALL, shuffle=True, augment=True)
    images, labels = next(iter(dataset))

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert labels.shape[0] == images.shape[0]


def test_build_datasets_returns_train_val_test(splits: Splits) -> None:
    datasets = build_datasets(splits, SMALL)

    assert set(datasets) == {"train", "val", "test"}
    for dataset in datasets.values():
        images, _ = next(iter(dataset))
        assert tuple(images.shape[1:]) == (32, 32, 3)
