from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

CLASS_FOLDERS: Mapping[str, str] = {
    "COVID": "COVID",
    "Normal": "Normal",
    "Viral Pneumonia": "Viral Pneumonia",
}
IMAGES_PER_CLASS = 20
IMAGE_SIZE = 48


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def synthetic_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def build_processed_dataset(root: Path) -> Path:
    for class_index, folder_name in enumerate(CLASS_FOLDERS.values()):
        for image_index in range(IMAGES_PER_CLASS):
            name = f"{folder_name}-{image_index}.png"
            write_image(
                root / folder_name / "images" / name,
                synthetic_image(class_index * 100 + image_index),
            )
    return root


def synthetic_mask() -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    margin = IMAGE_SIZE // 4
    mask[margin:-margin, margin:-margin] = 255
    return mask


def add_synthetic_masks(root: Path, class_folders=CLASS_FOLDERS) -> Path:
    mask = synthetic_mask()
    for folder_name in class_folders.values():
        images_dir = root / folder_name / "images"
        for image_path in images_dir.glob("*.png"):
            write_image(root / folder_name / "masks" / image_path.name, mask)
    return root
