from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tensorflow import keras

from covid_xray.config import CLASS_COLUMN, LABEL_TO_ID
from covid_xray.preprocessing import Splits
from covid_xray.transfer_learning import TransferConfig
from covid_xray.transfer_learning.dataset import (
    apply_lung_mask,
    build_augmentation_pipeline,
    build_dataset,
    build_datasets,
    compute_balanced_class_weights,
    crop_to_lung_bbox,
    load_cropped_lung_image,
    load_mask,
    load_mask_only,
    oversample_to_balance,
)

SMALL = TransferConfig(image_size=(32, 32), batch_size=4)
MASKED = TransferConfig(image_size=(32, 32), batch_size=4, mask_lungs=True)
MASKED_POOLING = TransferConfig(image_size=(32, 32), batch_size=4, masked_pooling=True)
MASK_ONLY = TransferConfig(image_size=(32, 32), batch_size=4, mask_only=True)
CROPPED = TransferConfig(image_size=(32, 32), batch_size=4, crop_lungs=True)
CROPPED_AUG = TransferConfig(
    image_size=(32, 32), batch_size=4, crop_lungs=True, random_translation=0.1, random_zoom=0.1
)
NO_FLIP = TransferConfig(image_size=(32, 32), batch_size=4, horizontal_flip=False)


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


def test_build_augmentation_pipeline_includes_flip_by_default() -> None:
    pipeline = build_augmentation_pipeline(horizontal_flip=True)

    assert any(isinstance(layer, keras.layers.RandomFlip) for layer in pipeline.layers)


def test_build_augmentation_pipeline_can_include_spatial_layers() -> None:
    pipeline = build_augmentation_pipeline(
        horizontal_flip=False, random_translation=0.1, random_zoom=0.1
    )

    layer_types = {type(layer) for layer in pipeline.layers}
    assert keras.layers.RandomTranslation in layer_types
    assert keras.layers.RandomZoom in layer_types


def test_build_augmentation_pipeline_can_disable_flip() -> None:
    pipeline = build_augmentation_pipeline(horizontal_flip=False)

    assert not any(isinstance(layer, keras.layers.RandomFlip) for layer in pipeline.layers)
    assert any(isinstance(layer, keras.layers.RandomRotation) for layer in pipeline.layers)


def test_crop_to_lung_bbox_trims_large_background() -> None:
    import tensorflow as tf

    image = tf.ones((64, 64, 3), dtype=tf.float32) * 100.0
    mask = tf.zeros((64, 64, 1), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[20, 20], [20, 40], [40, 20], [40, 40]],
        [[255.0], [255.0], [255.0], [255.0]],
    )

    cropped = crop_to_lung_bbox(image, mask, threshold=127, margin_fraction=0.05)

    assert cropped.shape[0] < image.shape[0]
    assert cropped.shape[1] < image.shape[1]


def test_build_dataset_with_crop_lungs_fills_frame(
    manifest_with_masks: pd.DataFrame,
) -> None:
    import tensorflow as tf

    dataset = build_dataset(manifest_with_masks, CROPPED, shuffle=False, augment=False)
    images, _ = next(iter(dataset))

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert float(tf.reduce_max(images)) > 0.0


def test_build_dataset_with_crop_lungs_applies_spatial_augment_on_train(
    manifest_with_masks: pd.DataFrame,
) -> None:
    from covid_xray.preprocessing import SplitConfig, split_manifest

    splits = split_manifest(manifest_with_masks, SplitConfig())
    datasets = build_datasets(splits, CROPPED_AUG)
    train_batches = list(datasets["train"].take(2))
    val_batches = list(datasets["val"].take(1))

    assert train_batches[0][0].shape[1:] == (32, 32, 3)
    assert val_batches[0][0].shape[1:] == (32, 32, 3)


def test_load_cropped_lung_image_returns_target_size(
    manifest_with_masks: pd.DataFrame,
) -> None:
    import tensorflow as tf

    row = manifest_with_masks.iloc[0]
    image = load_cropped_lung_image(
        tf.constant(row["image_path"]),
        tf.constant(row["mask_path"]),
        (32, 32),
        threshold=127,
        margin_fraction=0.08,
    )

    assert tuple(image.shape) == (32, 32, 3)


def test_build_dataset_without_horizontal_flip_keeps_shape(manifest: pd.DataFrame) -> None:
    dataset = build_dataset(manifest, NO_FLIP, shuffle=True, augment=True)
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


def test_load_mask_only_returns_rgb_mask(manifest_with_masks: pd.DataFrame) -> None:
    import tensorflow as tf

    mask_path = manifest_with_masks["mask_path"].iloc[0]

    mask_rgb = load_mask_only(tf.constant(mask_path), (32, 32))

    assert tuple(mask_rgb.shape) == (32, 32, 3)
    channels = mask_rgb.numpy()
    assert np.array_equal(channels[..., 0], channels[..., 1])
    assert np.array_equal(channels[..., 1], channels[..., 2])


def test_build_dataset_with_mask_only_ignores_original_image(
    manifest_with_masks: pd.DataFrame,
) -> None:
    dataset = build_dataset(manifest_with_masks, MASK_ONLY, shuffle=False, augment=False)
    images, labels = next(iter(dataset))

    values = np.unique(images.numpy())

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert labels.shape[0] == images.shape[0]
    assert set(values.tolist()) <= {0.0, 255.0}


def test_build_datasets_with_mask_only_returns_train_val_test(
    manifest_with_masks: pd.DataFrame,
) -> None:
    from covid_xray.preprocessing import SplitConfig, split_manifest

    splits_with_masks = split_manifest(manifest_with_masks, SplitConfig())
    datasets = build_datasets(splits_with_masks, MASK_ONLY)

    assert set(datasets) == {"train", "val", "test"}
    for dataset in datasets.values():
        images, _ = next(iter(dataset))
        assert tuple(images.shape[1:]) == (32, 32, 3)


def test_mask_lungs_mask_only_and_crop_lungs_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        TransferConfig(mask_lungs=True, mask_only=True)
    with pytest.raises(ValueError):
        TransferConfig(mask_lungs=True, crop_lungs=True)
    with pytest.raises(ValueError):
        TransferConfig(mask_only=True, crop_lungs=True)


def test_build_dataset_with_masked_pooling_yields_image_and_mask(
    manifest_with_masks: pd.DataFrame,
) -> None:
    dataset = build_dataset(manifest_with_masks, MASKED_POOLING, shuffle=False, augment=False)
    (images, masks), labels = next(iter(dataset))

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert tuple(masks.shape[1:]) == (32, 32, 1)
    assert images.shape[0] == masks.shape[0] == labels.shape[0]
    assert float(np.max(masks.numpy())) <= 255.0


def test_build_dataset_with_masked_pooling_and_mask_lungs_zeroes_background(
    manifest_with_masks: pd.DataFrame,
) -> None:
    config = TransferConfig(image_size=(32, 32), batch_size=4, masked_pooling=True, mask_lungs=True)
    dataset = build_dataset(manifest_with_masks, config, shuffle=False, augment=False)
    (images, _), _ = next(iter(dataset))

    corners = images.numpy()[:, 0, 0, :]

    assert np.all(corners == 0.0)


def _imbalance(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            manifest[manifest[CLASS_COLUMN] == "COVID"],
            manifest[manifest[CLASS_COLUMN] == "Normal"].iloc[:5],
            manifest[manifest[CLASS_COLUMN] == "Viral Pneumonia"].iloc[:10],
        ]
    ).reset_index(drop=True)


def test_oversample_to_balance_equalizes_class_counts(manifest: pd.DataFrame) -> None:
    balanced = oversample_to_balance(_imbalance(manifest), random_state=0)

    counts = balanced[CLASS_COLUMN].value_counts()

    assert counts.nunique() == 1
    assert int(counts["COVID"]) == 20


def test_oversample_to_balance_marks_added_rows_as_duplicates(
    manifest: pd.DataFrame,
) -> None:
    balanced = oversample_to_balance(_imbalance(manifest), random_state=0)

    covid_rows = balanced[balanced[CLASS_COLUMN] == "COVID"]
    normal_rows = balanced[balanced[CLASS_COLUMN] == "Normal"]
    viral_rows = balanced[balanced[CLASS_COLUMN] == "Viral Pneumonia"]

    assert not covid_rows["is_duplicate"].any()
    assert int(normal_rows["is_duplicate"].sum()) == 15
    assert int(viral_rows["is_duplicate"].sum()) == 10


def test_oversample_to_balance_is_deterministic(manifest: pd.DataFrame) -> None:
    imbalanced = _imbalance(manifest)

    first = oversample_to_balance(imbalanced, random_state=3)
    second = oversample_to_balance(imbalanced, random_state=3)

    pd.testing.assert_frame_equal(first, second)


def test_build_dataset_augments_only_duplicate_rows(manifest: pd.DataFrame) -> None:
    frame = manifest.iloc[:4].reset_index(drop=True).copy()
    frame["is_duplicate"] = [False, True, False, True]
    config = TransferConfig(image_size=(32, 32), batch_size=4, horizontal_flip=False)

    dataset = build_dataset(frame, config, shuffle=False, augment=False)
    reference = build_dataset(
        frame.drop(columns=["is_duplicate"]), config, shuffle=False, augment=False
    )

    images = next(iter(dataset))[0].numpy()
    reference_images = next(iter(reference))[0].numpy()

    assert np.array_equal(images[0], reference_images[0])
    assert np.array_equal(images[2], reference_images[2])
    assert not np.array_equal(images[1], reference_images[1])
    assert not np.array_equal(images[3], reference_images[3])


def test_build_dataset_marks_no_rows_as_duplicate_when_column_absent(
    manifest: pd.DataFrame,
) -> None:
    dataset = build_dataset(manifest, SMALL, shuffle=False, augment=False)
    images, labels = next(iter(dataset))

    assert tuple(images.shape[1:]) == (32, 32, 3)
    assert labels.shape[0] == images.shape[0]
