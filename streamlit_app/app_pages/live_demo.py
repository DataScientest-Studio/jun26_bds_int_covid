import streamlit as st
from PIL import Image

from lib.demo import (
    PRIMARY_MODEL_NAME,
    predict_with_explanation,
    preload_primary_model,
    read_uploaded_image,
)
from lib.paths import APP_ASSETS_DIR, DATA_DIR
from lib.ui import disclaimer, footer, page_intro


page_intro(
    "07 · Live demo",
    "One model, four prepared classes, one honest explanation",
    "The defense uses the fine-tuned EfficientNetB0 and keeps a static fallback ready.",
)

disclaimer()

prepared_examples = {
    "COVID": DATA_DIR / "processed" / "COVID" / "images" / "COVID-1002.png",
    "Lung Opacity · outside-lung attention example": DATA_DIR
    / "processed"
    / "Lung_Opacity"
    / "images"
    / "Lung_Opacity-100.png",
    "Normal": DATA_DIR / "processed" / "Normal" / "images" / "Normal-1005.png",
    "Viral Pneumonia": DATA_DIR
    / "processed"
    / "Viral Pneumonia"
    / "images"
    / "Viral Pneumonia-1003.png",
}

with st.container(horizontal=True):
    st.badge(PRIMARY_MODEL_NAME, icon=":material/model_training:", color="blue")
    st.badge("Preloads on page open", icon=":material/bolt:", color="green")
    st.badge("All four probabilities", icon=":material/bar_chart:", color="violet")

source = st.segmented_control(
    "Input source",
    ["Prepared examples", "Upload an X-ray", "Offline fallback"],
    default="Prepared examples",
    required=True,
    width="stretch",
)

selected_image: Image.Image | None = None
selected_name = ""

left, right = st.columns([1, 1.25], gap="large")
with left:
    if source == "Prepared examples":
        example_name = st.selectbox("Prepared class example", list(prepared_examples))
        example_path = prepared_examples[example_name]
        selected_image = Image.open(example_path).convert("L")
        selected_name = example_path.name
        st.image(selected_image, caption=f"{example_name} · {selected_name}")
        if "outside-lung" in example_name:
            st.warning(
                "This known example is useful for showing that Grad-CAM can concentrate away from the lungs.",
                icon=":material/visibility:",
            )
    elif source == "Upload an X-ray":
        uploaded_file = st.file_uploader(
            "Upload one frontal chest X-ray",
            type=["png", "jpg", "jpeg"],
            max_upload_size=20,
            help="PNG or JPEG, up to 20 MB. The image is converted to grayscale and resized for the model.",
        )
        if uploaded_file is not None:
            try:
                selected_image = read_uploaded_image(uploaded_file.getvalue())
                selected_name = uploaded_file.name
                st.image(selected_image, caption=selected_name)
            except ValueError as error:
                st.error(str(error), icon=":material/error:")
        else:
            st.caption("Choose one image; prepared examples remain available if upload fails.")
    else:
        fallback_path = APP_ASSETS_DIR / "live_demo_fallback.png"
        if fallback_path.exists():
            st.image(str(fallback_path), caption="Successful-result screenshot saved before the defense.")
        else:
            st.warning(
                "Fallback screenshot has not been generated yet.",
                icon=":material/image_not_supported:",
            )

with right:
    if source == "Offline fallback":
        with st.container(border=True):
            st.subheader("Fallback narration")
            st.write(
                "Use this screenshot if the live runtime fails. Point to the predicted class, all four probabilities, and the Grad-CAM overlay; then repeat that confidence is not clinical certainty."
            )
    elif selected_image is None:
        with st.container(border=True):
            st.subheader("Ready for one image")
            st.write("Select a prepared example or upload a frontal chest X-ray.")
    elif st.button("Run prediction", type="primary", icon=":material/play_arrow:"):
        with st.container(border=True):
            try:
                with st.spinner(f"Running {PRIMARY_MODEL_NAME}..."):
                    result = predict_with_explanation(selected_image)
                st.success(
                    f"Prediction: {result['label']} · confidence {result['confidence']:.1%}",
                    icon=":material/check_circle:",
                )
                probabilities, explanation = st.columns([1, 1.05], gap="medium")
                with probabilities:
                    st.bar_chart(
                        result["probabilities"],
                        x="Class",
                        y="Probability",
                        y_label="Model probability",
                        horizontal=True,
                    )
                with explanation:
                    st.image(
                        result["overlay"],
                        caption="Grad-CAM: warmer colors had more influence.",
                    )
                st.warning(
                    "Probabilities are model outputs, not disease probabilities. Grad-CAM is not lesion segmentation and may highlight non-lung regions.",
                    icon=":material/health_and_safety:",
                )
            except Exception as error:
                st.error(f"Prediction could not be completed: {error}", icon=":material/error:")

# Visiting the page warms the cached resource before the presenter clicks Run.
try:
    preload_primary_model()
except Exception as error:
    st.error(f"The primary model could not be preloaded: {error}", icon=":material/error:")

footer("Live demo")
