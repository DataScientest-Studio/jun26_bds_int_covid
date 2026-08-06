from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARRAY_DIR = DATA_DIR / "arrays"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
IMAGE_GLOB = "*.png"

CLASS_FOLDERS: Mapping[str, str] = MappingProxyType(
    {
        "COVID": "COVID",
        "Lung_Opacity": "Lung_Opacity",
        "Normal": "Normal",
        "Viral Pneumonia": "Viral Pneumonia",
    }
)
CLASS_NAMES: Tuple[str, ...] = tuple(CLASS_FOLDERS)
LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(CLASS_NAMES)}
)
ID_TO_LABEL: Mapping[int, str] = MappingProxyType(
    {index: name for name, index in LABEL_TO_ID.items()}
)
POSITIVE_CLASS = "COVID"

CLASS_COLUMN = "class"
IMAGE_PATH_COLUMN = "image_path"
MASK_PATH_COLUMN = "mask_path"
MANIFEST_COLUMNS: Tuple[str, ...] = (CLASS_COLUMN, IMAGE_PATH_COLUMN, MASK_PATH_COLUMN)

SPLITS: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
RANDOM_STATE = 42
