from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from ..config import (
    CLASS_COLUMN,
    IMAGE_PATH_COLUMN,
    IS_DUPLICATE_COLUMN,
    LABEL_TO_ID,
    MASK_PATH_COLUMN,
    RANDOM_STATE,
)
from ..preprocessing.manifest import Splits
from .config import TransferConfig

AUTOTUNE = tf.data.AUTOTUNE


def encode_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[CLASS_COLUMN].map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def oversample_to_balance(
    frame: pd.DataFrame,
    class_column: str = CLASS_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    target = int(frame[class_column].value_counts().max())

    balanced_parts = []
    for _, group in frame.groupby(class_column, sort=False):
        group = group.assign(**{IS_DUPLICATE_COLUMN: False})
        deficit = target - len(group)
        if deficit > 0:
            extra_positions = rng.choice(len(group), size=deficit, replace=True)
            extra = group.iloc[extra_positions].assign(**{IS_DUPLICATE_COLUMN: True})
            group = pd.concat([group, extra], ignore_index=True)
        balanced_parts.append(group)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def compute_balanced_class_weights(frame: pd.DataFrame) -> Dict[int, float]:
    labels = encode_labels(frame)
    class_ids = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=class_ids, y=labels
    )
    return dict(zip(class_ids.tolist(), weights.tolist()))


def load_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)
    image = tf.image.resize(image, image_size, method="area")
    return tf.image.grayscale_to_rgb(image)


def load_image_raw(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)
    return tf.image.grayscale_to_rgb(image)


def load_mask_raw(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    return tf.io.decode_png(raw, channels=1)


def align_mask_to_image(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    target_h = tf.shape(image)[0]
    target_w = tf.shape(image)[1]
    mask_h = tf.shape(mask)[0]
    mask_w = tf.shape(mask)[1]
    needs_resize = tf.logical_or(
        tf.not_equal(target_h, mask_h),
        tf.not_equal(target_w, mask_w),
    )

    def _resize() -> tf.Tensor:
        return tf.image.resize(mask, [target_h, target_w], method="nearest")

    return tf.cond(needs_resize, _resize, lambda: mask)


def crop_to_lung_bbox(
    image: tf.Tensor,
    mask: tf.Tensor,
    threshold: int,
    margin_fraction: float,
) -> tf.Tensor:
    binary = tf.cast(mask[..., 0] > threshold, tf.int32)
    coords = tf.where(binary)

    def _full_image() -> tf.Tensor:
        return image

    def _crop() -> tf.Tensor:
        y_min = tf.reduce_min(coords[:, 0])
        y_max = tf.reduce_max(coords[:, 0])
        x_min = tf.reduce_min(coords[:, 1])
        x_max = tf.reduce_max(coords[:, 1])
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        margin = tf.cast(
            tf.maximum(
                1.0,
                tf.cast(tf.maximum(height, width), tf.float32) * margin_fraction,
            ),
            tf.int64,
        )
        y1 = tf.maximum(tf.constant(0, dtype=tf.int64), y_min - margin)
        y2 = tf.minimum(tf.cast(tf.shape(image)[0], tf.int64), y_max + margin + 1)
        x1 = tf.maximum(tf.constant(0, dtype=tf.int64), x_min - margin)
        x2 = tf.minimum(tf.cast(tf.shape(image)[1], tf.int64), x_max + margin + 1)
        return image[y1:y2, x1:x2, :]

    return tf.cond(tf.greater(tf.shape(coords)[0], 0), _crop, _full_image)


def load_cropped_lung_image(
    path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: Tuple[int, int],
    threshold: int,
    margin_fraction: float,
) -> tf.Tensor:
    image = load_image_raw(path)
    mask = align_mask_to_image(image, load_mask_raw(mask_path))
    cropped = crop_to_lung_bbox(image, mask, threshold, margin_fraction)
    return tf.image.resize(cropped, image_size, method="area")


def load_mask(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    mask = tf.io.decode_png(raw, channels=1)
    return tf.image.resize(mask, image_size, method="nearest")


def load_mask_only(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Load a segmentation mask as a 3-channel model input, ignoring the X-ray pixels.

    Used to test how much class signal is carried by lung shape/geometry alone,
    with zero pixel-intensity information from the original image.
    """
    return tf.image.grayscale_to_rgb(load_mask(path, image_size))


def apply_lung_mask(image: tf.Tensor, mask: tf.Tensor, threshold: int) -> tf.Tensor:
    binary_mask = tf.cast(mask > threshold, image.dtype)
    return image * binary_mask


def _needs_mask_paths(config: TransferConfig) -> bool:
    return config.mask_lungs or config.mask_only or config.crop_lungs or config.masked_pooling


def _split_image_and_mask(combined: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return combined[..., :3], combined[..., 3:]


def _combine_image_and_mask(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    return tf.concat([image, tf.cast(mask, image.dtype)], axis=-1)


def build_augmentation_pipeline(
    horizontal_flip: bool = True,
    random_translation: float = 0.0,
    random_zoom: float = 0.0,
) -> keras.Sequential:
    layers = []
    if horizontal_flip:
        layers.append(keras.layers.RandomFlip("horizontal"))
    layers.append(keras.layers.RandomRotation(0.05))
    if random_translation > 0:
        layers.append(
            keras.layers.RandomTranslation(
                height_factor=random_translation,
                width_factor=random_translation,
                fill_mode="constant",
                fill_value=0.0,
            )
        )
    if random_zoom > 0:
        layers.append(
            keras.layers.RandomZoom(
                height_factor=(-random_zoom, random_zoom),
                width_factor=(-random_zoom, random_zoom),
                fill_mode="constant",
                fill_value=0.0,
            )
        )
    return keras.Sequential(layers)


def build_dataset(
    frame: pd.DataFrame,
    config: TransferConfig = TransferConfig(),
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    paths = frame[IMAGE_PATH_COLUMN].to_numpy()
    labels = encode_labels(frame)
    if IS_DUPLICATE_COLUMN in frame.columns:
        needs_augment = frame[IS_DUPLICATE_COLUMN].to_numpy()
    else:
        needs_augment = np.zeros(len(frame), dtype=bool)

    if _needs_mask_paths(config):
        mask_paths = frame[MASK_PATH_COLUMN].to_numpy()
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, mask_paths, labels, needs_augment)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels, needs_augment))

    if shuffle and len(frame) > 0:
        dataset = dataset.shuffle(
            buffer_size=len(frame), seed=config.random_state, reshuffle_each_iteration=True
        )

    if config.mask_only:
        dataset = dataset.map(
            lambda path, mask_path, label, needs_aug: (
                load_mask_only(mask_path, config.image_size),
                label,
                needs_aug,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    elif config.masked_pooling:
        dataset = dataset.map(
            lambda path, mask_path, label, needs_aug: (
                _combine_image_and_mask(
                    apply_lung_mask(
                        load_image(path, config.image_size),
                        load_mask(mask_path, config.image_size),
                        config.mask_threshold,
                    )
                    if config.mask_lungs
                    else load_image(path, config.image_size),
                    load_mask(mask_path, config.image_size),
                ),
                label,
                needs_aug,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    elif config.mask_lungs:
        dataset = dataset.map(
            lambda path, mask_path, label, needs_aug: (
                apply_lung_mask(
                    load_image(path, config.image_size),
                    load_mask(mask_path, config.image_size),
                    config.mask_threshold,
                ),
                label,
                needs_aug,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    elif config.crop_lungs:
        dataset = dataset.map(
            lambda path, mask_path, label, needs_aug: (
                load_cropped_lung_image(
                    path,
                    mask_path,
                    config.image_size,
                    config.mask_threshold,
                    config.crop_margin_fraction,
                ),
                label,
                needs_aug,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda path, label, needs_aug: (
                load_image(path, config.image_size),
                label,
                needs_aug,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.batch(config.batch_size)

    use_spatial_augment = config.crop_lungs and shuffle
    if augment or bool(needs_augment.any()) or use_spatial_augment:
        translation = config.random_translation if (config.crop_lungs or augment) else 0.0
        zoom = config.random_zoom if (config.crop_lungs or augment) else 0.0
        augmentation = build_augmentation_pipeline(
            horizontal_flip=config.horizontal_flip,
            random_translation=translation,
            random_zoom=zoom,
        )

        def _augment_batch(
            images: tf.Tensor, labels: tf.Tensor, needs_aug: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            augmented = augmentation(images, training=True)
            if augment or use_spatial_augment:
                output = augmented
            else:
                select_mask = tf.reshape(needs_aug, [-1, 1, 1, 1])
                output = tf.where(select_mask, augmented, images)
            if config.masked_pooling:
                image, mask = _split_image_and_mask(output)
                return (image, mask), labels
            return output, labels

        dataset = dataset.map(_augment_batch, num_parallel_calls=AUTOTUNE)
    else:
        if config.masked_pooling:

            def _split_batch(
                combined: tf.Tensor, labels: tf.Tensor, needs_aug: tf.Tensor
            ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
                image, mask = _split_image_and_mask(combined)
                return (image, mask), labels

            dataset = dataset.map(_split_batch, num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(
                lambda images, labels, needs_aug: (images, labels), num_parallel_calls=AUTOTUNE
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
