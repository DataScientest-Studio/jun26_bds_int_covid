from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from covid_xray.cnn import CNNConfig

# Tiny enough that the whole suite runs in seconds on CPU. Defined here rather
# than in conftest.py: `conftest` is not a unique module name across test
# directories, so importing from it collides with tests/training/conftest.py.
SMALL_CONFIG = CNNConfig(image_size=(16, 16), filters=(4, 8), batch_size=4, epochs=1)

CLASS_FOLDERS: Mapping[str, str] = {
    "COVID": "COVID",
    "Normal": "Normal",
    "Viral Pneumonia": "Viral Pneumonia",
}
IMAGES_PER_CLASS = 20
IMAGE_SIZE = 32
MASK_MARGIN = IMAGE_SIZE // 4


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def synthetic_image(seed: int) -> np.ndarray:
    # Every pixel is strictly positive, so a zero in the output can only have
    # come from masking and never from the source image itself.
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def synthetic_mask() -> np.ndarray:
    """A centred square 'lung field'.

    Unlike the all-255 mask in the training helpers, this one is partial, so
    the lungs and background conditions produce genuinely different tensors
    and the region logic is actually exercised.
    """
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[MASK_MARGIN:-MASK_MARGIN, MASK_MARGIN:-MASK_MARGIN] = 255
    return mask


def build_raw_dataset(root: Path) -> Path:
    mask = synthetic_mask()
    for class_index, folder_name in enumerate(CLASS_FOLDERS.values()):
        for image_index in range(IMAGES_PER_CLASS):
            name = f"{folder_name}-{image_index}.png"
            write_image(
                root / folder_name / "images" / name,
                synthetic_image(class_index * 100 + image_index),
            )
            write_image(root / folder_name / "masks" / name, mask)
    return root
