import streamlit as st

from lib.demo import MODEL_OPTIONS, predict_with_explanation, read_uploaded_image
from lib.ui import disclaimer, footer, page_intro


page_intro(
    "07 · Live demo",
    "Test one chest X-ray, then inspect where the model looked",
    "The output is a research-model prediction with uncertainty—not a diagnosis.",
)

disclaimer()

left, right = st.columns([1, 1.25], gap="large")
with left:
    model_name = st.selectbox("Model", list(MODEL_OPTIONS))
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray",
        type=["png", "jpg", "jpeg"],
        help="PNG or JPEG, up to 20 MB. The image is converted to grayscale and resized for the selected model.",
    )
    st.caption(
        "For a meaningful demonstration, use a frontal chest X-ray with similar framing to the training collection."
    )

    if uploaded_file is not None:
        try:
            uploaded_image = read_uploaded_image(uploaded_file.getvalue())
            st.image(uploaded_image, caption="Uploaded X-ray")
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
            uploaded_image = None
    else:
        uploaded_image = None

with right:
    if uploaded_image is None:
        with st.container(border=True):
            st.markdown("#### Ready for the live test")
            st.write(
                "Choose one of the two curated models, upload an image, and run inference. The model is loaded only when needed and then cached."
            )
            st.markdown(
                "**Output**\n\n"
                "- Predicted class and confidence\n"
                "- Probability for all four classes\n"
                "- Grad-CAM overlay for a qualitative explanation"
            )
    elif st.button("Run prediction", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Running inference and Grad-CAM..."):
                result = predict_with_explanation(uploaded_image, model_name)
            st.success(
                f"Prediction: {result['label']} · confidence {result['confidence']:.1%}",
                icon=":material/check_circle:",
            )
            st.bar_chart(
                result["probabilities"],
                x="Class",
                y="Probability",
                y_label="Model probability",
                horizontal=True,
            )
            st.image(
                result["overlay"],
                caption="Grad-CAM overlay: warmer colors indicate regions that most influenced this prediction.",
            )
            st.warning(
                "Do not interpret the heatmap as a lesion segmentation. Grad-CAM is coarse and can highlight spurious regions.",
                icon=":material/visibility:",
            )
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}", icon=":material/error:")

footer("Live demo")

