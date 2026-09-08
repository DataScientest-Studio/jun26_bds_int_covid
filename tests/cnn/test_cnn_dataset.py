from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cnn_helpers import IMAGE_SIZE, MASK_MARGIN, SMALL_CONFIG
from dataclasses import replace

from covid_xray.cnn import build_dataset, compute_class_weights
from covid_xray.config import MASK_PATH_COLUMN


def first_batch(frame: pd.DataFrame, region: str) -> np.ndarray:
    config = replace(SMALL_CONFIG, image_size=(IMAGE_SIZE, IMAGE_SIZE), region=region)
    dataset = build_dataset(frame, config)
    images, _ = next(iter(dataset))
    return images.numpy()


def test_full_region_keeps_every_pixel(manifest: pd.DataFrame) -> None:
    images = first_batch(manifest, "full")

    assert (images > 0).all()


def test_lungs_region_zeroes_the_outside(manifest: pd.DataFrame) -> None:
    images = first_batch(manifest, "lungs")
    inside = images[:, MASK_MARGIN:-MASK_MARGIN, MASK_MARGIN:-MASK_MARGIN, :]

    assert (inside > 0).all()
    assert images.sum() == pytest.approx(inside.sum())


def test_background_region_is_the_complement_of_lungs(manifest: pd.DataFrame) -> None:
    full = first_batch(manifest, "full")
    lungs = first_batch(manifest, "lungs")
    background = first_batch(manifest, "background")

    # Every pixel belongs to exactly one of the two regions, so the two
    # conditions must reconstruct the unmasked image.
    assert np.allclose(lungs + background, full)
    assert not np.allclose(background, 0)


def test_shapes_are_identical_across_regions(manifest: pd.DataFrame) -> None:
    shapes = {first_batch(manifest, region).shape for region in ("full", "lungs", "background")}

    assert len(shapes) == 1


def test_masked_region_requires_a_mask_column(manifest: pd.DataFrame) -> None:
    without_masks = manifest.drop(columns=[MASK_PATH_COLUMN])
    config = replace(SMALL_CONFIG, region="lungs")

    with pytest.raises(ValueError, match=MASK_PATH_COLUMN):
        build_dataset(without_masks, config)


def test_class_weights_are_inversely_proportional_to_frequency(manifest: pd.DataFrame) -> None:
    frame = pd.concat([manifest, manifest[manifest["class"] == "COVID"]])

    weights = compute_class_weights(frame)
    present = {k: v for k, v in weights.items() if v > 0}

    # COVID is now the majority class, so it must carry the smallest weight.
    assert min(present, key=present.get) == 0

def test_intensity_mode_handles_mismatched_image_and_mask_shapes(
    tmp_path,
):
    import numpy as np
    import tensorflow as tf

    from covid_xray.cnn.dataset import load_masked_image

    image = np.ones(
        (299, 299, 1),
        dtype=np.uint8,
    ) * 150

    mask = np.zeros(
        (256, 256, 1),
        dtype=np.uint8,
    )

    mask[40:220, 30:105, 0] = 255
    mask[40:220, 150:225, 0] = 255

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"

    tf.io.write_file(
        str(image_path),
        tf.io.encode_png(image),
    )

    tf.io.write_file(
        str(mask_path),
        tf.io.encode_png(mask),
    )

    result = load_masked_image(
        image_path=tf.constant(str(image_path)),
        mask_path=tf.constant(str(mask_path)),
        image_size=(128, 128),
        region="lungs",
        mask_threshold=127,
        lung_normalization="intensity",
    )

    result = result.numpy()

    assert result.shape == (128, 128, 1)
    assert np.isfinite(result).all()