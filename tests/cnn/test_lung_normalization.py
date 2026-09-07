import numpy as np
import covid_xray.cnn.lung_normalization as ln

from covid_xray.cnn.lung_normalization import (
    keep_two_largest_components,
    normalize_lung_geometry_pair,
)


def test_keep_two_largest_components():
    mask = np.zeros((50, 50), dtype=np.uint8)

    mask[5:20, 5:15] = 1
    mask[5:20, 30:40] = 1

    # Tiny artifact
    mask[45:47, 45:47] = 1

    cleaned = keep_two_largest_components(mask)

    assert cleaned[45:47, 45:47].sum() == 0


def test_geometry_output_shape():
    image = np.zeros((299, 299), dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)

    mask[50:200, 40:100] = 255
    mask[50:200, 150:210] = 255

    normalized, normalized_mask, angle = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=(128, 128),
        )
    )

    assert normalized.shape == (128, 128)
    assert normalized_mask.shape == (128, 128)


def test_no_pixels_outside_final_mask():
    image = np.ones(
        (256, 256),
        dtype=np.float32,
    ) * 200

    mask = np.zeros(
        (256, 256),
        dtype=np.uint8,
    )

    mask[40:220, 30:105] = 255
    mask[40:220, 150:225] = 255

    normalized, normalized_mask, _ = (
        normalize_lung_geometry_pair(
            image=image,
            mask=mask,
            target_size=(128, 128),
        )
    )

    assert np.all(
        normalized[normalized_mask == 0] == 0
    )

def test_extreme_rotation_is_rejected(monkeypatch):

    mask = np.zeros(
        (128, 128),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ln,
        "estimate_lung_rotation_raw",
        lambda _: 45.0,
    )

    angle = ln.estimate_lung_rotation(mask)

    assert angle == 0.0