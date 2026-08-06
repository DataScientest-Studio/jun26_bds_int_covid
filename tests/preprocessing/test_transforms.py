from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from helpers import IMAGE_SIZE, synthetic_image, synthetic_mask

from covid_xray.preprocessing import (
    PreprocessConfig,
    apply_clahe,
    apply_lung_mask,
    normalize_min_max,
    preprocess_xray,
    read_grayscale,
    resize_image,
    resize_mask,
)


def test_preprocess_xray_returns_normalized_target_size_image(
    image_and_mask: Tuple[Path, Path],
) -> None:
    image_path, mask_path = image_and_mask

    result = preprocess_xray(
        image_path, mask_path, PreprocessConfig(target_size=(32, 32))
    )

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert result.min() == pytest.approx(0.0, abs=1e-6)
    assert result.max() == pytest.approx(1.0, abs=1e-6)


def test_preprocess_xray_is_deterministic_without_augmentation(
    image_and_mask: Tuple[Path, Path],
) -> None:
    image_path, mask_path = image_and_mask
    config = PreprocessConfig(target_size=(32, 32))

    first = preprocess_xray(image_path, mask_path, config)
    second = preprocess_xray(image_path, mask_path, config)

    assert np.array_equal(first, second)


def test_preprocess_xray_applies_lung_mask(
    image_and_mask: Tuple[Path, Path],
) -> None:
    image_path, mask_path = image_and_mask

    result = preprocess_xray(
        image_path,
        mask_path,
        PreprocessConfig(target_size=(IMAGE_SIZE, IMAGE_SIZE), apply_lung_mask=True),
    )

    assert np.all(result[:16, :] == 0)
    assert np.all(result[48:, :] == 0)
    assert result[16:48, 16:48].max() > 0


def test_preprocess_xray_clahe_changes_pixels(
    image_and_mask: Tuple[Path, Path],
) -> None:
    image_path, mask_path = image_and_mask

    plain = preprocess_xray(image_path, mask_path, PreprocessConfig(target_size=(32, 32)))
    equalized = preprocess_xray(
        image_path, mask_path, PreprocessConfig(target_size=(32, 32), apply_clahe=True)
    )

    assert plain.shape == equalized.shape
    assert not np.allclose(plain, equalized)


def test_preprocess_xray_raises_for_unreadable_files(
    tmp_path: Path, image_and_mask: Tuple[Path, Path]
) -> None:
    image_path, mask_path = image_and_mask

    with pytest.raises(ValueError, match="Could not read image"):
        preprocess_xray(tmp_path / "missing.png", mask_path)

    with pytest.raises(ValueError, match="Could not read image"):
        preprocess_xray(image_path, tmp_path / "missing.png")


def test_normalize_min_max_scales_to_unit_range() -> None:
    image = np.array([[0, 128], [200, 255]], dtype=np.uint8)

    result = normalize_min_max(image)

    assert result.min() == pytest.approx(0.0, abs=1e-6)
    assert result.max() == pytest.approx(1.0, abs=1e-6)


def test_normalize_min_max_handles_constant_image() -> None:
    constant = np.full((8, 8), 120, dtype=np.uint8)

    result = normalize_min_max(constant)

    assert result.dtype == np.float32
    assert np.all(result == 0)


def test_apply_clahe_preserves_shape_and_dtype() -> None:
    image = synthetic_image(1)

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_apply_lung_mask_zeroes_outside_mask() -> None:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE), 200, dtype=np.uint8)

    result = apply_lung_mask(image, synthetic_mask())

    assert result[0, 0] == 0
    assert result[32, 32] == 200


def test_resize_helpers_keep_mask_values_binary() -> None:
    resized_image = resize_image(synthetic_image(2), (32, 32))
    resized_mask = resize_mask(synthetic_mask(), (32, 32))

    assert resized_image.shape == (32, 32)
    assert set(np.unique(resized_mask)).issubset({0, 255})


def test_read_grayscale_returns_two_dimensional_array(
    image_and_mask: Tuple[Path, Path],
) -> None:
    image = read_grayscale(image_and_mask[0])

    assert image.ndim == 2
    assert image.dtype == np.uint8
