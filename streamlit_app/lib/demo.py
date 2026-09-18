from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from covid_xray.config import CLASS_NAMES
from covid_xray.explainability.gradcam import (
    build_gradcam_models,
    compute_gradcam_heatmap,
    overlay_heatmap,
    resize_heatmap,
)
from lib.paths import MODELS_DIR


MATERIALIZED_MODEL_PATH = MODELS_DIR / "transfer_efficientnetb0_lung_only_materialized.keras"
MATERIALIZED_SCAN_ONLY = "Materialized EfficientNetB0 · scan only"
MATERIALIZED_WITH_MASK = "Materialized EfficientNetB0 · scan + mask"

MODEL_OPTIONS = {
    "Simple CNN": MODELS_DIR / "cnn_simple.keras",
    "EfficientNetB0": MODELS_DIR / "transfer_efficientnetb0.keras",
    "Fine-tuned EfficientNetB0": MODELS_DIR / "transfer_efficientnetb0_finetuned.keras",
    MATERIALIZED_SCAN_ONLY: MATERIALIZED_MODEL_PATH,
    MATERIALIZED_WITH_MASK: MATERIALIZED_MODEL_PATH,
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    return keras.models.load_model(model_path, compile=False)


def read_uploaded_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        return Image.open(BytesIO(content)).convert("L")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable PNG or JPEG image.") from error


def requires_lung_mask(model_name: str) -> bool:
    return model_name == MATERIALIZED_WITH_MASK


def prepare_image(
    image: Image.Image,
    model,
    lung_mask: Image.Image | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple):
        raise ValueError("This demo supports single-image-input models only.")
    height, width, channels = input_shape[1:]
    if lung_mask is None:
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        grayscale = np.asarray(resized, dtype=np.float32)
    else:
        grayscale_source = np.asarray(image, dtype=np.uint8)
        mask_source = np.asarray(lung_mask, dtype=np.uint8)
        if mask_source.shape != grayscale_source.shape:
            mask_source = cv2.resize(
                mask_source,
                (grayscale_source.shape[1], grayscale_source.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        binary_mask = mask_source > 127
        masked = np.where(binary_mask, grayscale_source, 0).astype(np.uint8)
        grayscale = cv2.resize(masked, (width, height), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(
            binary_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        grayscale[resized_mask == 0] = 0
        grayscale = grayscale.astype(np.float32)
    model_input = grayscale[..., np.newaxis]
    if channels == 3:
        model_input = np.repeat(model_input, 3, axis=-1)
    return model_input, grayscale


def predict_with_explanation(
    image: Image.Image,
    model_name: str,
    lung_mask: Image.Image | None = None,
) -> dict:
    if requires_lung_mask(model_name) and lung_mask is None:
        raise ValueError("The scan + mask model requires a paired lung mask.")
    model = load_model(str(MODEL_OPTIONS[model_name]))
    model_input, grayscale = prepare_image(image, model, lung_mask=lung_mask)
    probabilities = model.predict(model_input[np.newaxis, ...], verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    grad_model, classifier_model = build_gradcam_models(model)
    heatmap, _ = compute_gradcam_heatmap(
        model_input,
        grad_model,
        classifier_model,
        pred_index=predicted_index,
    )
    heatmap = resize_heatmap(heatmap, grayscale.shape)
    overlay = overlay_heatmap(grayscale, heatmap, alpha=0.42)
    probability_frame = pd.DataFrame(
        {
            "Class": [name.replace("_", " ") for name in CLASS_NAMES],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)
    return {
        "label": CLASS_NAMES[predicted_index].replace("_", " "),
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probability_frame,
        "overlay": overlay,
    }
