from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..config import (
    CLASS_COLUMN,
    IMAGE_PATH_COLUMN,
    LABEL_TO_ID,
    MASK_PATH_COLUMN,
)
from ..preprocessing.manifest import Splits
from .config import CNNConfig
from .lung_normalization import normalize_lung_input

AUTOTUNE = tf.data.AUTOTUNE


def encode_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[CLASS_COLUMN].map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def load_masked_image(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: Tuple[int, int],
    region: str,
    mask_threshold: int,
    lung_normalization: str = "none",
) -> tf.Tensor:
    """Load an X-ray and apply the requested image region."""

    raw = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        raw,
        channels=1,
    )

    if region == "full":
        image = tf.image.resize(
            image,
            image_size,
            method="area",
        )

        return tf.cast(
            image,
            tf.float32,
        )

    mask_raw = tf.io.read_file(mask_path)

    mask = tf.io.decode_png(
        mask_raw,
        channels=1,
    )

    if region == "lungs" and lung_normalization != "none":

        def _normalize(
            image_np: np.ndarray,
            mask_np: np.ndarray,
        ) -> np.ndarray:
            return normalize_lung_input(
                image=image_np,
                mask=mask_np,
                target_size=image_size,
                mode=lung_normalization,
                mask_threshold=mask_threshold,
            )

        normalized = tf.numpy_function(
            func=_normalize,
            inp=[image, mask],
            Tout=tf.float32,
        )

        normalized.set_shape(
            (
                image_size[0],
                image_size[1],
                1,
            )
        )

        return normalized

    image = tf.image.resize(
        image,
        image_size,
        method="area",
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    mask = tf.image.resize(
        mask,
        image_size,
        method="nearest",
    )

    mask = tf.cast(
        mask,
        tf.float32,
    )

    is_lung = mask > float(mask_threshold)

    if region == "lungs":
        keep = is_lung

    elif region == "background":
        keep = tf.logical_not(is_lung)

    else:
        raise ValueError(
            f"Unsupported region: {region!r}"
        )

    return image * tf.cast(
        keep,
        tf.float32,
    )


def build_augmentation_pipeline(seed: int) -> keras.Sequential:
    """Geometric augmentation applied AFTER masking.

    Because the mask has already been composited into the image, the transform
    moves image and mask together by construction -- there is no pairing bug to
    worry about here, unlike the numpy augment_pair() path.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, fill_mode="constant", seed=seed),
            keras.layers.RandomZoom(0.10, fill_mode="constant", seed=seed),
        ],
        name="augmentation",
    )


def build_dataset(
    frame: pd.DataFrame,
    config: CNNConfig = CNNConfig(),
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    if config.region != "full" and MASK_PATH_COLUMN not in frame.columns:
        raise ValueError(
            f"region={config.region!r} needs a {MASK_PATH_COLUMN!r} column in the manifest"
        )

    image_paths = frame[IMAGE_PATH_COLUMN].to_numpy()
    mask_paths = (
        frame[MASK_PATH_COLUMN].to_numpy()
        if MASK_PATH_COLUMN in frame.columns
        else image_paths  # unused when region == "full"
    )
    labels = encode_labels(frame)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))
    if shuffle and len(frame) > 0:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=config.random_state,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, mask_path, label: (
            load_masked_image(
                image_path,
                mask_path,
                config.image_size,
                config.region,
                config.mask_threshold,
                config.lung_normalization,
            ),
            label,
        ),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(config.batch_size)

    if augment:
        augmentation = build_augmentation_pipeline(config.random_state)
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.prefetch(AUTOTUNE)


def build_datasets(splits: Splits, config: CNNConfig = CNNConfig()) -> Dict[str, tf.data.Dataset]:
    return {
        "train": build_dataset(splits.train, config, shuffle=True, augment=config.augment),
        "val": build_dataset(splits.val, config, shuffle=False, augment=False),
        "test": build_dataset(splits.test, config, shuffle=False, augment=False),
    }


def compute_class_weights(frame: pd.DataFrame) -> Dict[int, float]:
    """Inverse-frequency weights, matching sklearn's class_weight="balanced".

    Computed on the training split only. Deriving them from the full manifest
    would leak the test class distribution into training.
    """
    labels = encode_labels(frame)
    n_classes = len(LABEL_TO_ID)
    counts = np.bincount(labels, minlength=n_classes)
    weights: Dict[int, float] = {}
    for class_id, count in enumerate(counts):
        # Absent classes get weight 0; they contribute no gradient either way.
        weights[class_id] = float(len(labels) / (n_classes * count)) if count else 0.0
    return weights
