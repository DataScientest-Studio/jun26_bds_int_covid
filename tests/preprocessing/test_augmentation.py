from __future__ import annotations

import numpy as np
import pytest
from helpers import synthetic_image, synthetic_mask

from covid_xray.preprocessing import (
    AugmentConfig,
    augment_pair,
    make_rng,
    random_brightness_contrast,
    random_flip,
    random_rotation,
    random_rotation_no_zoom,
    random_translation,
    random_zoom,
    rotation_zoom_scale,
)


@pytest.mark.parametrize(
    "operation",
    [
        random_flip,
        random_rotation,
        random_rotation_no_zoom,
        random_zoom,
        random_translation,
    ],
)
def test_geometric_augmentations_preserve_shapes(operation) -> None:
    image = synthetic_image(3)
    mask = synthetic_mask()

    augmented_image, augmented_mask = operation(image, mask)

    assert augmented_image.shape == image.shape
    assert augmented_mask.shape == mask.shape


@pytest.mark.parametrize("zoom_range", [0.0, 0.001, 0.05, 0.5])
def test_random_zoom_handles_every_zoom_factor(zoom_range: float) -> None:
    image = synthetic_image(6)
    mask = synthetic_mask()

    zoomed_image, zoomed_mask = random_zoom(
        image, mask, zoom_range=zoom_range, rng=make_rng(1)
    )

    assert zoomed_image.shape == image.shape
    assert zoomed_mask.shape == mask.shape


def test_random_flip_mirrors_horizontally() -> None:
    image = synthetic_image(7)

    flipped, _ = random_flip(image, synthetic_mask())

    assert np.array_equal(flipped, image[:, ::-1])


def test_random_brightness_contrast_stays_in_byte_range() -> None:
    image = synthetic_image(8)

    result = random_brightness_contrast(image, rng=make_rng(2))

    assert result.dtype == np.uint8
    assert result.shape == image.shape


def test_rotation_zoom_scale_grows_with_angle() -> None:
    assert rotation_zoom_scale(100, 100, 0) == pytest.approx(1.0)
    assert rotation_zoom_scale(100, 100, 15) > 1.0
    assert rotation_zoom_scale(100, 100, 30) > rotation_zoom_scale(100, 100, 15)


def test_rotation_with_zoom_avoids_reflected_mask_borders() -> None:
    mask = synthetic_mask()

    _, rotated_mask = random_rotation(synthetic_image(9), mask, rng=make_rng(3))

    assert set(np.unique(rotated_mask)).issubset({0, 255})


def test_augment_pair_is_reproducible_with_seeded_rng() -> None:
    image = synthetic_image(4)
    mask = synthetic_mask()

    first = augment_pair(image.copy(), mask.copy(), rng=make_rng(42))
    second = augment_pair(image.copy(), mask.copy(), rng=make_rng(42))
    other = augment_pair(image.copy(), mask.copy(), rng=make_rng(43))

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_augment_pair_can_be_disabled_by_zero_probabilities() -> None:
    image = synthetic_image(5)
    mask = synthetic_mask()
    config = AugmentConfig(
        flip_probability=0.0,
        rotation_probability=0.0,
        zoom_probability=0.0,
        translation_probability=0.0,
        brightness_contrast_probability=0.0,
    )

    augmented_image, augmented_mask = augment_pair(
        image.copy(), mask.copy(), config=config, rng=make_rng(0)
    )

    assert np.array_equal(augmented_image, image)
    assert np.array_equal(augmented_mask, mask)


def test_augment_pair_keeps_shapes_across_many_draws() -> None:
    image = synthetic_image(10)
    mask = synthetic_mask()
    rng = make_rng(123)

    for _ in range(25):
        augmented_image, augmented_mask = augment_pair(
            image.copy(), mask.copy(), rng=rng
        )
        assert augmented_image.shape == image.shape
        assert augmented_mask.shape == mask.shape
