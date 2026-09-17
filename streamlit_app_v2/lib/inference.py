from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from covid_xray.explainability.gradcam import gradcam_for_image

from .paths import CNN_DIR, MODELS_DIR, TRANSFER_DIR


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: Path
    metrics: Path
    size: tuple[int, int]
    channels: int


MODELS = (
    ModelSpec(
        "cnn_scratch",
        "Scratch CNN · highest saved test score",
        MODELS_DIR / "cnn_scratch.keras",
        CNN_DIR / "cnn_scratch_metrics.json",
        (128, 128),
        1,
    ),
    ModelSpec(
        "cnn_simple",
        "Simple CNN · full image",
        MODELS_DIR / "cnn_simple.keras",
        CNN_DIR / "cnn_simple_metrics.json",
        (128, 128),
        1,
    ),
    ModelSpec(
        "transfer_finetuned",
        "EfficientNetB0 · fine-tuned",
        MODELS_DIR / "transfer_efficientnetb0_finetuned.keras",
        TRANSFER_DIR / "transfer_efficientnetb0_finetuned_metrics.json",
        (224, 224),
        3,
    ),
    ModelSpec(
        "transfer_frozen",
        "EfficientNetB0 · frozen backbone",
        MODELS_DIR / "transfer_efficientnetb0.keras",
        TRANSFER_DIR / "transfer_efficientnetb0_metrics.json",
        (224, 224),
        3,
    ),
)


def available_models() -> tuple[ModelSpec, ...]:
    return tuple(model for model in MODELS if model.path.exists())


@lru_cache(maxsize=6)
def load_model(path: str):
    return keras.models.load_model(path, compile=False)


def read_image(uploaded_file) -> np.ndarray:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("L")
    return np.asarray(image, dtype=np.uint8)


def prepare(image: np.ndarray, spec: ModelSpec) -> np.ndarray:
    resized = cv2.resize(image, spec.size, interpolation=cv2.INTER_AREA).astype(np.float32)
    if spec.channels == 1:
        return resized[..., np.newaxis]
    return np.repeat(resized[..., np.newaxis], 3, axis=-1)


def predict(model, tensor: np.ndarray) -> np.ndarray:
    return model.predict(tensor[np.newaxis, ...], verbose=0)[0]


def explain(model, tensor: np.ndarray) -> dict:
    return gradcam_for_image(model, tensor)

