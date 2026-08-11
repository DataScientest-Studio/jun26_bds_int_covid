from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..config import CLASS_COLUMN, IMAGE_PATH_COLUMN, LABEL_TO_ID
from ..preprocessing.dataset import encode_labels
from ..preprocessing.transforms import read_grayscale, resize_image
from .config import BaselineConfig

PIXEL_MAX_VALUE = 255.0


def load_flat_features(
    frame: pd.DataFrame, config: BaselineConfig = BaselineConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    n_features = config.image_size[0] * config.image_size[1]
    if frame.empty:
        return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=np.int64)

    vectors = []
    for image_path in frame[IMAGE_PATH_COLUMN]:
        image = read_grayscale(image_path)
        image = resize_image(image, config.image_size)
        vectors.append(image.flatten().astype(np.float32) / PIXEL_MAX_VALUE)

    X = np.stack(vectors)
    y = encode_labels(frame[CLASS_COLUMN].to_numpy(), LABEL_TO_ID)
    return X, y
