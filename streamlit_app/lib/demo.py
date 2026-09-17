from io import BytesIO
from pathlib import Path

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


MODEL_OPTIONS = {
    "CNN from scratch": MODELS_DIR / "cnn_scratch.keras",
    "Fine-tuned EfficientNetB0": MODELS_DIR / "transfer_efficientnetb0_finetuned.keras",
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


def prepare_image(image: Image.Image, model) -> tuple[np.ndarray, np.ndarray]:
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple):
        raise ValueError("This demo supports single-image-input models only.")
    height, width, channels = input_shape[1:]
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    grayscale = np.asarray(resized, dtype=np.float32)
    model_input = grayscale[..., np.newaxis]
    if channels == 3:
        model_input = np.repeat(model_input, 3, axis=-1)
    return model_input, grayscale


def predict_with_explanation(image: Image.Image, model_name: str) -> dict:
    model = load_model(str(MODEL_OPTIONS[model_name]))
    model_input, grayscale = prepare_image(image, model)
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

