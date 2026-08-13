from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import CLASS_COLUMN, IMAGE_PATH_COLUMN, LABEL_TO_ID, MASK_PATH_COLUMN
from ..preprocessing.dataset import encode_labels
from ..preprocessing.transforms import read_grayscale, resize_image, resize_mask
from .config import BaselineConfig

PIXEL_MAX_VALUE = 255.0

def apply_region(
    image: np.ndarray,
    mask: np.ndarray,
    region: str,
    threshold: int = 127,
) -> np.ndarray:
    """Keep only the pixels belonging to `region`, zero out the rest.

    The discarded region is set to 0 rather than removed, so every image keeps
    the same flattened length and the feature index of a pixel stays comparable
    across the three conditions.
    """
    if region == "full":
        return image

    is_lung = mask > threshold
    keep = is_lung if region == "lungs" else ~is_lung
    return image * keep  # bool broadcasts as 1/0

def load_flat_features(
    frame: pd.DataFrame, config: BaselineConfig = BaselineConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten each X-ray to a 1-D vector of scaled pixel values.

    Identical for every region except the masking step, so the only thing that
    varies between the full / lungs / background runs is which pixels survive.
    """
    n_features = config.image_size[0] * config.image_size[1]
    if frame.empty:
        return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=np.int64)

    needs_mask = config.region != "full"
    if needs_mask and MASK_PATH_COLUMN not in frame.columns:
        raise ValueError(
            f"region={config.region!r} needs a {MASK_PATH_COLUMN!r} column in the manifest"
        )

    vectors = []
    for row in frame.itertuples(index=False):
        image = read_grayscale(getattr(row, IMAGE_PATH_COLUMN))
        image = resize_image(image, config.image_size)

        if needs_mask:
            mask_path = getattr(row, MASK_PATH_COLUMN)
            if not Path(mask_path).exists():
                raise FileNotFoundError(f"Missing mask for region masking: {mask_path}")
            mask = read_grayscale(mask_path)
            # INTER_NEAREST keeps the mask strictly binary; INTER_AREA would
            # blur the lung boundary into intermediate greys and leak a halo.
            mask = resize_mask(mask, config.image_size)
            image = apply_region(image, mask, config.region, config.mask_threshold)

        vectors.append(image.flatten().astype(np.float32) / PIXEL_MAX_VALUE)

    X = np.stack(vectors)
    y = encode_labels(frame[CLASS_COLUMN].to_numpy(), LABEL_TO_ID)
    return X, y
