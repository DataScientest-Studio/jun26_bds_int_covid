from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .augmentation import AugmentConfig, augment_pair
from .config import PreprocessConfig

NORMALIZATION_EPSILON = 1e-8


def read_grayscale(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_lung_mask(
    image: np.ndarray, mask: np.ndarray, threshold: int = 127
) -> np.ndarray:
    binary_mask = (mask > threshold).astype(np.uint8)
    return cv2.bitwise_and(image, image, mask=binary_mask)


def normalize_min_max(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    minimum, maximum = image.min(), image.max()
    return (image - minimum) / (maximum - minimum + NORMALIZATION_EPSILON)


def preprocess_xray(
    image_path: Path | str,
    mask_path: Path | str,
    config: PreprocessConfig = PreprocessConfig(),
    augment_config: AugmentConfig = AugmentConfig(),
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    image = read_grayscale(image_path)
    mask = read_grayscale(mask_path)

    image = resize_image(image, config.target_size)
    mask = resize_mask(mask, config.target_size)

    if config.augment:
        image, mask = augment_pair(image, mask, config=augment_config, rng=rng)

    if config.apply_clahe:
        image = apply_clahe(
            image,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )

    if config.apply_lung_mask:
        image = apply_lung_mask(image, mask, threshold=config.mask_threshold)

    return normalize_min_max(image)
