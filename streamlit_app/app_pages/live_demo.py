import streamlit as st

from lib.demo import (
    MODEL_OPTIONS,
    predict_with_explanation,
    read_uploaded_image,
    requires_lung_mask,
)
from lib.ui import disclaimer, footer, page_intro


page_intro(
    "07 · Live demo",
    "Test multiple chest X-rays, then inspect where the model looked",
    "Compare five inference options across a batch of images—the outputs are not diagnoses.",
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

    uploaded_mask_files = []
    if requires_lung_mask(model_name):
        st.info(
            "This option applies the training-time lungs-only transform. Upload one matching lung mask per X-ray, in the same order.",
            icon=":material/masks:",
        )
        uploaded_mask_files = st.file_uploader(
            "Upload paired lung masks",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            max_upload_size=20,
            help="Masks are paired with X-rays by upload order and thresholded at 127.",
            key="lung_masks",
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

    uploaded_masks = []
    for uploaded_mask_file in uploaded_mask_files:
        try:
            uploaded_masks.append(read_uploaded_image(uploaded_mask_file.getvalue()))
        except ValueError as error:
            st.error(f"{uploaded_mask_file.name}: {error}", icon=":material/error:")

    masks_ready = not requires_lung_mask(model_name) or (
        len(uploaded_masks) == len(uploaded_images)
    )
    if requires_lung_mask(model_name) and uploaded_images and not masks_ready:
        st.warning(
            f"Upload {len(uploaded_images)} readable lung mask{'s' if len(uploaded_images) != 1 else ''}; "
            f"{len(uploaded_masks)} {'are' if len(uploaded_masks) != 1 else 'is'} ready.",
            icon=":material/warning:",
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
    elif not masks_ready:
        with st.container(border=True):
            st.markdown("#### Paired masks required")
            st.write(
                "Add one lung segmentation mask for each uploaded X-ray. Files are paired by their upload order."
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
                        lung_mask = (
                            uploaded_masks[index - 1]
                            if requires_lung_mask(model_name)
                            else None
                        )
                        result = predict_with_explanation(
                            item["image"], model_name, lung_mask=lung_mask
                        )
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
