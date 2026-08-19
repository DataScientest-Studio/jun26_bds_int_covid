from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..config import CLASS_COLUMN, IMAGE_PATH_COLUMN, LABEL_TO_ID, MASK_PATH_COLUMN
from ..preprocessing.manifest import Splits
from .config import TransferConfig

AUTOTUNE = tf.data.AUTOTUNE


def encode_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[CLASS_COLUMN].map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def load_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)
    image = tf.image.resize(image, image_size, method="area")
    return tf.image.grayscale_to_rgb(image)


def load_mask(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    mask = tf.io.decode_png(raw, channels=1)
    return tf.image.resize(mask, image_size, method="nearest")


def apply_lung_mask(image: tf.Tensor, mask: tf.Tensor, threshold: int) -> tf.Tensor:
    binary_mask = tf.cast(mask > threshold, image.dtype)
    return image * binary_mask


def build_augmentation_pipeline() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
        ]
    )


def build_dataset(
    frame: pd.DataFrame,
    config: TransferConfig = TransferConfig(),
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    paths = frame[IMAGE_PATH_COLUMN].to_numpy()
    labels = encode_labels(frame)

    if config.mask_lungs:
        mask_paths = frame[MASK_PATH_COLUMN].to_numpy()
        dataset = tf.data.Dataset.from_tensor_slices((paths, mask_paths, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle and len(frame) > 0:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=config.random_state, reshuffle_each_iteration=True
        )

    if config.mask_lungs:
        dataset = dataset.map(
            lambda path, mask_path, label: (
                apply_lung_mask(
                    load_image(path, config.image_size),
                    load_mask(mask_path, config.image_size),
                    config.mask_threshold,
                ),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda path, label: (load_image(path, config.image_size), label),
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.batch(config.batch_size)

    if augment:
        augmentation = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.prefetch(AUTOTUNE)


def build_datasets(
    splits: Splits, config: TransferConfig = TransferConfig()
) -> Dict[str, tf.data.Dataset]:
    return {
        "train": build_dataset(splits.train, config, shuffle=True, augment=config.augment),
        "val": build_dataset(splits.val, config, shuffle=False, augment=False),
        "test": build_dataset(splits.test, config, shuffle=False, augment=False),
    }
