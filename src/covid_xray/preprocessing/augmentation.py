from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import RANDOM_STATE

Pair = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class AugmentConfig:
    flip_probability: float = 0.5
    rotation_probability: float = 0.5
    max_rotation_degrees: float = 15.0
    rotation_extra_zoom: float = 0.02
    zoom_probability: float = 0.5
    zoom_range: float = 0.08
    translation_probability: float = 0.5
    shift_range: float = 0.05
    brightness_contrast_probability: float = 0.4
    brightness_range: float = 0.1
    contrast_range: float = 0.1


def make_rng(seed: Optional[int] = RANDOM_STATE) -> random.Random:
    return random.Random(seed)


def _resolve_rng(rng: Optional[random.Random]) -> random.Random:
    return rng if rng is not None else random


def random_flip(image: np.ndarray, mask: np.ndarray) -> Pair:
    return cv2.flip(image, 1), cv2.flip(mask, 1)


def rotation_zoom_scale(width: int, height: int, angle_degrees: float) -> float:
    angle_radians = np.deg2rad(abs(angle_degrees))
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    bounding_width = width * cos_a + height * sin_a
    bounding_height = width * sin_a + height * cos_a
    return float(max(bounding_width / width, bounding_height / height))


def random_rotation_no_zoom(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: float = AugmentConfig.max_rotation_degrees,
    rng: Optional[random.Random] = None,
) -> Pair:
    rng = _resolve_rng(rng)
    angle = rng.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
        flags=cv2.INTER_NEAREST,
    )
    return rotated_image, rotated_mask


def random_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: float = AugmentConfig.max_rotation_degrees,
    extra_zoom: float = AugmentConfig.rotation_extra_zoom,
    rng: Optional[random.Random] = None,
) -> Pair:
    rng = _resolve_rng(rng)
    angle = rng.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    scale = rotation_zoom_scale(width, height, angle) * (
        1 + rng.uniform(0, extra_zoom)
    )
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    rotated_image = cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def random_zoom(
    image: np.ndarray,
    mask: np.ndarray,
    zoom_range: float = AugmentConfig.zoom_range,
    rng: Optional[random.Random] = None,
) -> Pair:
    rng = _resolve_rng(rng)
    height, width = image.shape[:2]
    zoom = 1 + rng.uniform(-zoom_range, zoom_range)
    new_height, new_width = int(height * zoom), int(width * zoom)

    zoomed_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    zoomed_mask = cv2.resize(
        mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    if new_height >= height and new_width >= width:
        top = (new_height - height) // 2
        left = (new_width - width) // 2
        zoomed_image = zoomed_image[top : top + height, left : left + width]
        zoomed_mask = zoomed_mask[top : top + height, left : left + width]
    else:
        pad_height, pad_width = height - new_height, width - new_width
        top, left = pad_height // 2, pad_width // 2
        zoomed_image = cv2.copyMakeBorder(
            zoomed_image,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_REFLECT,
        )
        zoomed_mask = cv2.copyMakeBorder(
            zoomed_mask,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return zoomed_image, zoomed_mask


def random_translation(
    image: np.ndarray,
    mask: np.ndarray,
    shift_range: float = AugmentConfig.shift_range,
    rng: Optional[random.Random] = None,
) -> Pair:
    rng = _resolve_rng(rng)
    height, width = image.shape[:2]
    shift_x = rng.uniform(-shift_range, shift_range) * width
    shift_y = rng.uniform(-shift_range, shift_range) * height
    matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    shifted_image = cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )
    shifted_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
        flags=cv2.INTER_NEAREST,
    )
    return shifted_image, shifted_mask


def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: float = AugmentConfig.brightness_range,
    contrast_range: float = AugmentConfig.contrast_range,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    rng = _resolve_rng(rng)
    alpha = 1 + rng.uniform(-contrast_range, contrast_range)
    beta = rng.uniform(-brightness_range, brightness_range) * 255
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: AugmentConfig = AugmentConfig(),
    rng: Optional[random.Random] = None,
) -> Pair:
    rng = _resolve_rng(rng)

    if rng.random() < config.flip_probability:
        image, mask = random_flip(image, mask)

    if rng.random() < config.rotation_probability:
        image, mask = random_rotation(
            image,
            mask,
            max_angle=config.max_rotation_degrees,
            extra_zoom=config.rotation_extra_zoom,
            rng=rng,
        )

    if rng.random() < config.zoom_probability:
        image, mask = random_zoom(image, mask, zoom_range=config.zoom_range, rng=rng)

    if rng.random() < config.translation_probability:
        image, mask = random_translation(
            image, mask, shift_range=config.shift_range, rng=rng
        )

    if rng.random() < config.brightness_contrast_probability:
        image = random_brightness_contrast(
            image,
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range,
            rng=rng,
        )

    return image, mask
