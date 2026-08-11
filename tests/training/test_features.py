from __future__ import annotations

import numpy as np
import pandas as pd

from covid_xray.config import LABEL_TO_ID
from covid_xray.training import BaselineConfig, load_flat_features


def test_load_flat_features_returns_flattened_scaled_vectors(manifest: pd.DataFrame) -> None:
    config = BaselineConfig(image_size=(8, 8))

    X, y = load_flat_features(manifest, config)

    assert X.shape == (len(manifest), 64)
    assert X.dtype == np.float32
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert y.shape == (len(manifest),)
    assert set(y.tolist()) <= set(LABEL_TO_ID.values())


def test_load_flat_features_handles_empty_frame() -> None:
    empty = pd.DataFrame(columns=["class", "image_path", "mask_path"])
    config = BaselineConfig(image_size=(8, 8))

    X, y = load_flat_features(empty, config)

    assert X.shape == (0, 64)
    assert y.shape == (0,)
