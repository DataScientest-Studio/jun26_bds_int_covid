import streamlit as st
from PIL import Image

from lib.demo import (
    PRIMARY_MODEL_NAME,
    predict_with_explanation,
    preload_primary_model,
    read_uploaded_image,
)
from lib.paths import APP_ASSETS_DIR, example_image
from lib.ui import disclaimer, footer, page_intro


page_intro(
    "07 · Live demo",
    "One model, four prepared classes, one honest explanation",
    "The defense uses the fine-tuned EfficientNetB0 and keeps a static fallback ready.",
)

disclaimer()

prepared_examples = {
    "COVID": example_image("COVID", "COVID-1002.png"),
    "Lung Opacity · outside-lung attention example": example_image(
        "Lung_Opacity", "Lung_Opacity-100.png"
    ),
    "Normal": example_image("Normal", "Normal-1005.png"),
    "Viral Pneumonia": example_image("Viral Pneumonia", "Viral Pneumonia-1003.png"),
}

DEFAULT_EXAMPLE = "Lung Opacity · outside-lung attention example"
MAX_GRID_COLS = 2

SelectedItem = tuple[str, Image.Image, str]


def chunked(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_prepared_items(names: list[str]) -> list[SelectedItem]:
    loaded: list[SelectedItem] = []
    for name in names:
        example_path = prepared_examples[name]
        if example_path.exists():
            loaded.append(
                (name, Image.open(example_path).convert("L"), example_path.name)
            )
        else:
            st.warning(f"Example image not found: {example_path.name}")
    return loaded


def render_image_grid(items: list[SelectedItem]) -> None:
    if not items:
        return
    st.subheader("Selected X-rays")
    for row in chunked(items, MAX_GRID_COLS):
        columns = st.columns(len(row), gap="medium")
        for column, (label, image, filename) in zip(columns, row):
            with column:
                st.image(image, caption=f"{label} · {filename}")


def render_result_card(label: str, filename: str, result: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{label}** · `{filename}`")
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


def render_result_grid(items: list[SelectedItem], results: list[dict]) -> None:
    st.subheader("Predictions")
    for row in chunked(list(zip(items, results)), MAX_GRID_COLS):
        columns = st.columns(len(row), gap="medium")
        for column, ((label, _, filename), result) in zip(columns, row):
            with column:
                render_result_card(label, filename, result)


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

selected_items: list[SelectedItem] = []

if source == "Prepared examples":
    chosen = st.pills(
        "Prepared class examples",
        list(prepared_examples),
        default=[DEFAULT_EXAMPLE],
        selection_mode="multi",
    )
    if chosen:
        selected_items = load_prepared_items(list(chosen))
        if any("outside-lung" in name for name in chosen):
            st.warning(
                "The Lung Opacity example is useful for showing that Grad-CAM can concentrate away from the lungs.",
                icon=":material/visibility:",
            )
    else:
        st.caption("Select one or more prepared examples to preview them together.")
elif source == "Upload an X-ray":
    uploaded_files = st.file_uploader(
        "Upload frontal chest X-rays",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        max_upload_size=20,
        help="PNG or JPEG, up to 20 MB each. Images are converted to grayscale and resized for the model.",
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                selected_items.append(
                    (
                        uploaded_file.name,
                        read_uploaded_image(uploaded_file.getvalue()),
                        uploaded_file.name,
                    )
                )
            except ValueError as error:
                st.error(f"{uploaded_file.name}: {error}", icon=":material/error:")
    else:
        st.caption("Choose one or more images; prepared examples remain available if upload fails.")
else:
    fallback_path = APP_ASSETS_DIR / "live_demo_fallback.png"
    if fallback_path.exists():
        st.image(str(fallback_path), caption="Successful-result screenshot saved before the defense.")
    else:
        st.warning(
            "Fallback screenshot has not been generated yet.",
            icon=":material/image_not_supported:",
        )

if source != "Offline fallback":
    render_image_grid(selected_items)

    if not selected_items:
        with st.container(border=True):
            st.subheader("Ready for images")
            st.write("Select one or more prepared examples or upload frontal chest X-rays.")
    else:
        image_count = len(selected_items)
        run_label = (
            "Run prediction"
            if image_count == 1
            else f"Run predictions on {image_count} images"
        )
        if st.button(run_label, type="primary", icon=":material/play_arrow:"):
            with st.spinner(f"Running {PRIMARY_MODEL_NAME}..."):
                successful_items: list[SelectedItem] = []
                results: list[dict] = []
                for label, image, filename in selected_items:
                    try:
                        results.append(predict_with_explanation(image))
                        successful_items.append((label, image, filename))
                    except Exception as error:
                        st.error(
                            f"{label} · {filename}: {error}",
                            icon=":material/error:",
                        )
                if results:
                    render_result_grid(successful_items, results)
                    st.warning(
                        "Probabilities are model outputs, not disease probabilities. Grad-CAM is not lesion segmentation and may highlight non-lung regions.",
                        icon=":material/health_and_safety:",
                    )
else:
    with st.container(border=True):
        st.subheader("Fallback narration")
        st.write(
            "Use this screenshot if the live runtime fails. Point to the predicted class, all four probabilities, and the Grad-CAM overlay; then repeat that confidence is not clinical certainty."
        )

try:
    preload_primary_model()
except Exception as error:
    st.error(f"The primary model could not be preloaded: {error}", icon=":material/error:")

footer("Live demo")
