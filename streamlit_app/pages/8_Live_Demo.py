import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st

from lib.inference import (
    available_demo_models,
    format_probability,
    load_demo_model,
    prepare_model_input,
    predict_proba,
    run_gradcam,
)
from lib.metrics import load_metrics_json, split_summary
from lib.inference import read_uploaded_image
from lib.ui import app_footer, configure_page, disclaimer_banner, page_header

configure_page("Live Demo")

page_header(
    "Live Demo",
    "Upload a chest X-ray, pick a saved model, and inspect the predicted class "
    "with a Grad-CAM overlay. Educational use only.",
)

disclaimer_banner()

models = available_demo_models()
if not models:
    st.error("No saved models found under models/. Train and save models before using the live demo.")
    st.stop()

model_labels = {spec.label: spec for spec in models}
selected_label = st.selectbox("Model", list(model_labels.keys()))
selected = model_labels[selected_label]

uploaded = st.file_uploader(
    "Chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Grayscale or RGB PNG/JPEG. The image is resized to the model input size.",
)

metrics = load_metrics_json(selected.metrics_path)
metrics_summary = split_summary(metrics, split="test")
if metrics_summary:
    metric_cols = st.columns(3, gap="small")
    metric_cols[0].metric("Saved model test accuracy", f"{100 * metrics_summary['accuracy']:.2f}%")
    metric_cols[1].metric("Saved model macro F1", f"{metrics_summary['macro_f1']:.3f}")
    metric_cols[2].metric(
        "Input size",
        f"{selected.image_size[0]}×{selected.image_size[1]}",
        f"{selected.channels}-channel",
    )

if uploaded is None:
    st.caption("Upload an image to run inference. Predictions are not medical advice.")
    app_footer()
    st.stop()

@st.cache_resource(show_spinner="Loading model…")
def cached_model(path: str):
    return load_demo_model(path)


model = cached_model(str(selected.model_path))

try:
    image = read_uploaded_image(uploaded)
    tensor = prepare_model_input(image, selected.image_size, selected.channels)
    probabilities = predict_proba(model, tensor)
    gradcam_result = run_gradcam(model, tensor)
except Exception as error:
    st.error(f"Inference failed: {error}")
    app_footer()
    st.stop()

predicted_label = gradcam_result["predicted_label"]
predicted_index = gradcam_result["predicted_index"]

left, right = st.columns([2, 3], gap="medium")
with left:
    st.markdown("**Input X-ray**")
    st.image(image, use_container_width=True)
with right:
    st.markdown("**Grad-CAM overlay**")
    st.image(gradcam_result["overlay"], use_container_width=True)
    st.caption(
        f"Predicted class: **{predicted_label}** "
        f"({format_probability(probabilities[predicted_index])} confidence). "
        "Heatmap shows where the model focused when making this prediction; "
        "it is not a clinical explanation."
    )

prob_rows = []
for index, class_name in enumerate(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]):
    prob_rows.append(
        {
            "Class": class_name,
            "Probability": format_probability(probabilities[index]),
            "Score": probabilities[index],
        }
    )

prob_frame = pd.DataFrame(prob_rows)
st.bar_chart(prob_frame.set_index("Class")[["Score"]], height=220)

st.dataframe(
    prob_frame[["Class", "Probability"]],
    hide_index=True,
    width="stretch",
    height=180,
)

st.caption(
    "Models exposed here are full-image classifiers trained on the project's "
    "stratified split. Uploads outside that distribution may behave unpredictably. "
    "For the causal shortcut analysis, see Interpretability & Limitations."
)

app_footer()
