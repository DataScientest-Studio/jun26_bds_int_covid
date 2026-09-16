from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from covid_xray.config import CLASS_NAMES, ID_TO_LABEL
from covid_xray.explainability.gradcam import gradcam_for_image

from .paths import CNN_REPORTS_DIR, MODELS_DIR, TRANSFER_REPORTS_DIR


@dataclass(frozen=True)
class DemoModelSpec:
    key: str
    label: str
    model_path: Path
    metrics_path: Path
    image_size: Tuple[int, int]
    channels: int


DEMO_MODELS: Tuple[DemoModelSpec, ...] = (
    DemoModelSpec(
        key="cnn_scratch",
        label="Scratch CNN (best test accuracy)",
        model_path=MODELS_DIR / "cnn_scratch.keras",
        metrics_path=CNN_REPORTS_DIR / "cnn_scratch_metrics.json",
        image_size=(128, 128),
        channels=1,
    ),
    DemoModelSpec(
        key="cnn_simple",
        label="Simple CNN (full image)",
        model_path=MODELS_DIR / "cnn_simple.keras",
        metrics_path=CNN_REPORTS_DIR / "cnn_simple_metrics.json",
        image_size=(128, 128),
        channels=1,
    ),
    DemoModelSpec(
        key="transfer_finetuned",
        label="EfficientNetB0 fine-tuned",
        model_path=MODELS_DIR / "transfer_efficientnetb0_finetuned.keras",
        metrics_path=TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_finetuned_metrics.json",
        image_size=(224, 224),
        channels=3,
    ),
    DemoModelSpec(
        key="transfer_frozen",
        label="EfficientNetB0 frozen backbone",
        model_path=MODELS_DIR / "transfer_efficientnetb0.keras",
        metrics_path=TRANSFER_REPORTS_DIR / "transfer_efficientnetb0_metrics.json",
        image_size=(224, 224),
        channels=3,
    ),
)


def demo_model_by_key(key: str) -> Optional[DemoModelSpec]:
    for spec in DEMO_MODELS:
        if spec.key == key:
            return spec
    return None


def available_demo_models() -> Tuple[DemoModelSpec, ...]:
    return tuple(spec for spec in DEMO_MODELS if spec.model_path.exists())


@lru_cache(maxsize=8)
def load_demo_model(model_path: str) -> keras.Model:
    return keras.models.load_model(model_path, compile=False)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    raw = uploaded_file.read()
    image = Image.open(BytesIO(raw)).convert("L")
    return np.array(image, dtype=np.uint8)


def prepare_model_input(
    image: np.ndarray,
    image_size: Tuple[int, int],
    channels: int,
) -> np.ndarray:
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    if channels == 1:
        tensor = resized.astype(np.float32)[..., np.newaxis]
    else:
        tensor = np.repeat(resized.astype(np.float32)[..., np.newaxis], 3, axis=-1)
    return tensor


def predict_proba(model: keras.Model, tensor: np.ndarray) -> np.ndarray:
    batch = tensor[np.newaxis, ...]
    return model.predict(batch, verbose=0)[0]


def run_gradcam(model: keras.Model, tensor: np.ndarray) -> dict:
    return gradcam_for_image(model, tensor)


def format_probability(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def class_color(class_name: str) -> str:
    from .ui import CLASS_COLORS

    return CLASS_COLORS.get(class_name, "#4c51bf")
