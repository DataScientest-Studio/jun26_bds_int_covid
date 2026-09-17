import streamlit as st

from lib.demo import MODEL_OPTIONS, predict_with_explanation, read_uploaded_image
from lib.ui import disclaimer, footer, page_intro


page_intro(
    "07 · Live demo",
    "Test multiple chest X-rays, then inspect where the model looked",
    "Compare three research models across a batch of images—the outputs are not diagnoses.",
)

disclaimer()

left, right = st.columns([1, 1.25], gap="large")
with left:
    model_name = st.segmented_control(
        "Model",
        list(MODEL_OPTIONS),
        default="Fine-tuned EfficientNetB0",
        required=True,
        width="stretch",
    )
    uploaded_files = st.file_uploader(
        "Upload chest X-rays",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        max_upload_size=20,
        help="Select one or more PNG or JPEG files, up to 20 MB each. Images are converted to grayscale and resized for the selected model.",
    )
    st.caption(
        "For a meaningful demonstration, use a frontal chest X-ray with similar framing to the training collection."
    )

    uploaded_images = []
    for uploaded_file in uploaded_files:
        try:
            uploaded_images.append(
                {
                    "name": uploaded_file.name,
                    "image": read_uploaded_image(uploaded_file.getvalue()),
                }
            )
        except ValueError as error:
            st.error(f"{uploaded_file.name}: {error}", icon=":material/error:")

    if uploaded_images:
        st.caption(f"{len(uploaded_images)} image{'s' if len(uploaded_images) != 1 else ''} ready")
        st.image(
            [item["image"] for item in uploaded_images],
            caption=[item["name"] for item in uploaded_images],
            width=140,
        )

with right:
    if not uploaded_images:
        with st.container(border=True):
            st.markdown("#### Ready for the live test")
            st.write(
                "Choose a model, upload one or more images, and run the batch. Models are loaded only when needed and then cached."
            )
            st.markdown(
                "**For every image**\n\n"
                "- Predicted class and confidence\n"
                "- Probability for all four classes\n"
                "- Grad-CAM overlay for a qualitative explanation"
            )
    elif st.button(
        f"Run {len(uploaded_images)} prediction{'s' if len(uploaded_images) != 1 else ''}",
        type="primary",
        icon=":material/play_arrow:",
    ):
        for index, item in enumerate(uploaded_images, start=1):
            with st.container(border=True):
                st.markdown(f"#### {index}. {item['name']}")
                try:
                    with st.spinner(f"Running {model_name}..."):
                        result = predict_with_explanation(item["image"], model_name)
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
                except Exception as error:
                    st.error(f"Prediction could not be completed: {error}", icon=":material/error:")

        st.warning(
            "Do not interpret the heatmaps as lesion segmentations. Grad-CAM is coarse and can highlight spurious regions.",
            icon=":material/visibility:",
        )

footer("Live demo")
