from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covid_xray.config import CLASS_COLUMN, LABEL_TO_ID
from covid_xray.preprocessing import Splits
from covid_xray.transfer_learning import TransferConfig
from covid_xray.transfer_learning.dataset import (
    apply_lung_mask,
    build_dataset,
    build_datasets,
    compute_balanced_class_weights,
    load_mask,
)

SMALL = TransferConfig(image_size=(32, 32), batch_size=4)
MASKED = TransferConfig(image_size=(32, 32), batch_size=4, mask_lungs=True)


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


def test_apply_lung_mask_zeroes_pixels_outside_mask() -> None:
    import tensorflow as tf

    image = tf.ones((4, 4, 3), dtype=tf.float32) * 200.0
    mask = tf.constant(
        [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]], dtype=tf.float32
    )
    mask = mask[..., tf.newaxis]

    masked = apply_lung_mask(image, mask, threshold=127)

    assert float(tf.reduce_sum(masked[0, 0])) == 0.0
    assert float(tf.reduce_sum(masked[1, 1])) == 600.0


def test_compute_balanced_class_weights_favors_minority_classes() -> None:
    frame = pd.DataFrame(
        {CLASS_COLUMN: ["Normal"] * 30 + ["COVID"] * 10 + ["Viral Pneumonia"] * 5}
    )

    weights = compute_balanced_class_weights(frame)

    assert set(weights) == {
        LABEL_TO_ID[name] for name in ("Normal", "COVID", "Viral Pneumonia")
    }
    assert weights[LABEL_TO_ID["Viral Pneumonia"]] > weights[LABEL_TO_ID["COVID"]]
    assert weights[LABEL_TO_ID["COVID"]] > weights[LABEL_TO_ID["Normal"]]


def test_compute_balanced_class_weights_uniform_for_balanced_frame(
    manifest: pd.DataFrame,
) -> None:
    weights = compute_balanced_class_weights(manifest)

    assert set(weights) == {LABEL_TO_ID[name] for name in manifest[CLASS_COLUMN].unique()}
    for class_name in manifest[CLASS_COLUMN].unique():
        assert weights[LABEL_TO_ID[class_name]] == pytest.approx(1.0)


def test_load_mask_resizes_to_target_size(manifest_with_masks: pd.DataFrame) -> None:
    import tensorflow as tf

    mask_path = manifest_with_masks["mask_path"].iloc[0]

    mask = load_mask(tf.constant(mask_path), (32, 32))

    assert tuple(mask.shape) == (32, 32, 1)


def test_build_dataset_with_mask_lungs_zeroes_background(
    manifest_with_masks: pd.DataFrame,
) -> None:
    dataset = build_dataset(manifest_with_masks, MASKED, shuffle=False, augment=False)
    images, _ = next(iter(dataset))

    corners = images.numpy()[:, 0, 0, :]

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert np.all(corners == 0.0)
