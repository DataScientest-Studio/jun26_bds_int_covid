from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


EPSILON = 1e-6

def align_image_to_mask_frame(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bring the X-ray into the same pixel coordinate system as the mask.

    In this dataset masks may be stored at 256x256 while the original
    X-rays can have other resolutions (e.g. 299x299).
    """
    image = np.squeeze(image).astype(np.float32)
    mask = np.squeeze(mask)

    if image.shape != mask.shape:
        mask_h, mask_w = mask.shape

        image = cv2.resize(
            image,
            (mask_w, mask_h),
            interpolation=cv2.INTER_AREA,
        )

    return image, mask

def _binary_mask(mask: np.ndarray, threshold: int) -> np.ndarray:
    mask = np.squeeze(mask)
    return (mask > threshold).astype(np.uint8)


def estimate_lung_rotation(
    binary_mask: np.ndarray,
    max_angle: float = 10.0,
    reject_angle: float = 20.0,
) -> float:
    raw_angle = estimate_lung_rotation_raw(
        binary_mask
    )

    # Extremely large estimates are likely unreliable.
    # Leave these images unrotated.
    if abs(raw_angle) > reject_angle:
        return 0.0

    return float(
        np.clip(
            raw_angle,
            -max_angle,
            max_angle,
        )
    )


def rotate_pair(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape

    center = (
        w / 2.0,
        h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, rotated_mask


def crop_to_lungs(
    image: np.ndarray,
    mask: np.ndarray,
    margin_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image, mask

    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)

    margin = max(
        1,
        int(max(height, width) * margin_fraction),
    )

    y1 = max(
        0,
        int(ys.min()) - margin,
    )

    y2 = min(
        image.shape[0],
        int(ys.max()) + margin + 1,
    )

    x1 = max(
        0,
        int(xs.min()) - margin,
    )

    x2 = min(
        image.shape[1],
        int(xs.max()) + margin + 1,
    )

    return (
        image[y1:y2, x1:x2],
        mask[y1:y2, x1:x2],
    )


def square_pad_and_resize(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    side = max(h, w)

    image_canvas = np.zeros(
        (side, side),
        dtype=np.float32,
    )

    mask_canvas = np.zeros(
        (side, side),
        dtype=np.uint8,
    )

    y0 = (side - h) // 2
    x0 = (side - w) // 2

    image_canvas[
        y0:y0 + h,
        x0:x0 + w,
    ] = image

    mask_canvas[
        y0:y0 + h,
        x0:x0 + w,
    ] = mask

    # OpenCV expects (width, height)
    output_size = (
        target_size[1],
        target_size[0],
    )

    image_resized = cv2.resize(
        image_canvas,
        output_size,
        interpolation=cv2.INTER_AREA,
    )

    mask_resized = cv2.resize(
        mask_canvas,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )

    mask_resized = (
        mask_resized > 0
    ).astype(np.uint8)

    return image_resized, mask_resized


def resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    output_size = (
        target_size[1],
        target_size[0],
    )

    image = cv2.resize(
        image,
        output_size,
        interpolation=cv2.INTER_AREA,
    )

    mask = cv2.resize(
        mask,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )

    mask = (mask > 0).astype(np.uint8)

    return image, mask


def normalize_lung_intensity(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Robust per-image lung normalization.

    Returns values in [0, 255] because the CNN already contains
    keras.layers.Rescaling(1/255).
    """
    lung = mask > 0

    if not np.any(lung):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    pixels = image[lung].astype(np.float32)

    p01, p99 = np.percentile(
        pixels,
        [1.0, 99.0],
    )

    if p99 - p01 < EPSILON:
        result = image.astype(np.float32)
        result[~lung] = 0.0
        return result

    clipped = np.clip(
        image.astype(np.float32),
        p01,
        p99,
    )

    lung_pixels = clipped[lung]

    mean = float(lung_pixels.mean())
    std = float(lung_pixels.std())

    if std < EPSILON:
        std = 1.0

    normalized = (
        clipped - mean
    ) / std

    # Avoid extreme outliers.
    normalized = np.clip(
        normalized,
        -3.0,
        3.0,
    )

    # Keep the CNN input convention: [0, 255].
    normalized = (
        (normalized + 3.0)
        / 6.0
        * 255.0
    )

    normalized[~lung] = 0.0

    return normalized.astype(np.float32)


def normalize_lung_input(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
    mode: str,
    mask_threshold: int = 127,
) -> np.ndarray:

    image = np.squeeze(image).astype(np.float32)

    binary_mask = _binary_mask(
        mask,
        threshold=mask_threshold,
    )

    if mode in {"geometry", "both"}:

        image, binary_mask, _ = geometry_normalize_pair(
            image=image,
            mask=mask,
            target_size=target_size,
            mask_threshold=mask_threshold,
        )

    else:
        # IMPORTANT: mask before resizing here too.
        image = (
            image
            * binary_mask.astype(np.float32)
        )

        image, binary_mask = resize_pair(
            image,
            binary_mask,
            target_size=target_size,
        )

        image[
            binary_mask == 0
        ] = 0.0

    if mode in {"intensity", "both"}:
        image = normalize_lung_intensity(
            image,
            binary_mask,
        )

    # Final safety mask.
    image[
        binary_mask == 0
    ] = 0.0

    return image[
        ..., np.newaxis
    ].astype(np.float32)

def normalize_lung_geometry_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    mask_threshold: int = 127,
) -> tuple[np.ndarray, np.ndarray, float]:

    return geometry_normalize_pair(
        image=image,
        mask=mask,
        target_size=target_size,
        mask_threshold=mask_threshold,
    )

def geometry_normalize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
    mask_threshold: int = 127,
) -> tuple[np.ndarray, np.ndarray, float]:

    # 1. First put image and mask into the SAME coordinate frame.
    image, mask = align_image_to_mask_frame(
        image,
        mask,
    )

    # 2. Binarize mask.
    mask = _binary_mask(
        mask,
        threshold=mask_threshold,
    )

    mask = keep_two_largest_components(mask)
    # Sanity check.
    if image.shape != mask.shape:
        raise ValueError(
            "Image/mask alignment failed: "
            f"image={image.shape}, mask={mask.shape}"
        )

    # 3. Remove non-lung information BEFORE geometry operations.
    lung_image = (
        image
        * mask.astype(np.float32)
    )

    # 4. Estimate orientation from the mask.
    angle = estimate_lung_rotation(
        mask
    )

    # 5. Rotate lung image and mask using exactly the same transform.
    lung_image, rotated_mask = rotate_pair(
        lung_image,
        mask,
        angle,
    )

    # Remove interpolation spill.
    lung_image[
        rotated_mask == 0
    ] = 0.0

    # 6. Crop according to rotated lung mask.
    lung_image, rotated_mask = crop_to_lungs(
        lung_image,
        rotated_mask,
    )

    # 7. Square-pad and resize.
    lung_image, rotated_mask = square_pad_and_resize(
        lung_image,
        rotated_mask,
        target_size=target_size,
    )

    # 8. Final hard mask.
    lung_image[
        rotated_mask == 0
    ] = 0.0

    return (
        lung_image.astype(np.float32),
        rotated_mask.astype(np.uint8),
        angle,
    )

def estimate_lung_rotation_raw(
    binary_mask: np.ndarray,
) -> float:
    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(1, n_labels):
        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area > 0:
            components.append(
                (area, centroids[label])
            )

    if len(components) < 2:
        return 0.0

    components.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    c1 = components[0][1]
    c2 = components[1][1]

    if c1[0] > c2[0]:
        c1, c2 = c2, c1

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    if abs(dx) < EPSILON:
        return 0.0

    return float(
        np.degrees(
            np.arctan2(dy, dx)
        )
    )

def keep_two_largest_components(
    mask: np.ndarray,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    # Empty or single-component mask: leave unchanged.
    if n_labels <= 2:
        return binary

    components = []

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        components.append((area, label))

    components.sort(reverse=True)

    keep_labels = [
        label
        for _, label in components[:2]
    ]

    cleaned = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    for label in keep_labels:
        cleaned[labels == label] = 1

    return cleaned