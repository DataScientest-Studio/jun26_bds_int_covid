import json

import pandas as pd
import streamlit as st

from .inference import available_models, explain, load_model, predict, prepare, read_image
from .metrics import class_score, comparison, percent, points, summary
from .paths import BASELINE_DIR, CNN_DIR, DATA_DIR, LATEX_FIGURES_DIR, TRANSFER_DIR
from .ui import (
    callout,
    disclaimer,
    evidence_header,
    fit_page,
    header,
    image,
    metric_chart,
    numbers,
    probability_chart,
    source,
    trio,
    verdict,
)


CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
FULL = TRANSFER_DIR / "transfer_efficientnetb0_metrics.json"
MASKED = TRANSFER_DIR / "transfer_efficientnetb0_masked_metrics.json"
MASK_ONLY = TRANSFER_DIR / "transfer_efficientnetb0_mask_only_metrics.json"


def _metric_cards(items):
    cols = st.columns(len(items), gap="medium", vertical_alignment="center")
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def opening():
    fit_page("cover")
    best = summary(CNN_DIR / "cnn_scratch_metrics.json")
    st.markdown('<div class="deck-kicker">Data science defense / chest radiography</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="deck-title"><h1>Can a model be <span class="accent">right</span><br>'
        "for the wrong reason?</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="deck-subtitle">A four-class COVID-19 X-ray classifier, followed from a '
        "promising score to the dataset shortcut hiding underneath it.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="byline">Berfin Aktas & Mert &nbsp; / &nbsp; Final project defense</div>',
        unsafe_allow_html=True,
    )
    numbers(
        [
            ("21,165", "raw chest X-rays"),
            ("4", "diagnostic classes"),
            ("13+", "trained experiments"),
            (percent(best["accuracy"], 2), "best saved test accuracy"),
        ]
    )
    disclaimer()
    source("Performance is loaded live from reports/cnn/cnn_scratch_metrics.json.")


def question():
    header(
        "01 / The question",
        "Accuracy is an answer. Trust is a second question.",
        "The project began as classification and became an investigation into what the model actually learned.",
    )
    trio(
        "Four visually overlapping conditions make chest X-ray classification difficult.",
        "Build progressively stronger models, then challenge their evidence with region-specific experiments.",
        "The best in-distribution score was strong, but removing the background exposed a class-specific failure.",
    )
    st.markdown(
        '<div class="thesis">Our real result is the gap between <em>prediction</em> and '
        "<em>explanation</em>.</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("01", "Audit", "Find source and encoding clues before training."),
        ("02", "Control", "Deduplicate, stratify and freeze the split."),
        ("03", "Benchmark", "Start with models too simple to hide behind."),
        ("04", "Escalate", "Train CNN and transfer-learning variants."),
        ("05", "Interrogate", "Remove regions and measure what survives."),
    ]
    cards = "".join(
        f'<div class="path-step"><div class="path-num">{number}</div>'
        f'<div class="path-title">{title}</div><div class="path-copy">{copy}</div></div>'
        for number, title, copy in steps
    )
    st.markdown(f'<div class="pathway">{cards}</div>', unsafe_allow_html=True)


def dataset():
    evidence_header(
        "02 / The material",
        "One dataset. Four classes. Several origins.",
        "The labels look unified in a folder tree, but the images were assembled from different repositories and acquisition pipelines.",
    )
    trio(
        "Source repository can become a proxy for diagnosis when each class has a different provenance.",
        "Audit balance and provenance before asking a model to learn the labels.",
        "Class imbalance and source-class dependence were visible before training began.",
    )
    image(
        LATEX_FIGURES_DIR / "class_distribution.png",
        "21,165 raw images. A stratified split preserves this imbalance; it does not remove source confounding.",
        width="stretch",
    )
    source("Class-distribution figure from reports/latex/figures/.")


def audit():
    evidence_header(
        "03 / The audit",
        "The first model was the dataset audit.",
        "Technical irregularities became early warnings of the shortcut-learning problem tested later.",
    )
    trio(
        "Leakage and encoding artifacts can inflate held-out performance without clinical signal.",
        "Run integrity checks, pHash duplicate detection, color-mode checks and intensity tests.",
        "59 redundant files, 140 class-confounded RGB images and significant intensity differences.",
    )
    duplicates, encoding, intensity = st.tabs(["Duplicates", "Encoding clue", "Intensity clue"])
    with duplicates:
        image(
            LATEX_FIGURES_DIR / "duplicate_distribution.png",
            "Duplicate groups were resolved before the 70/15/15 split.",
            width="stretch",
        )
    with encoding:
        image(
            LATEX_FIGURES_DIR / "viral_pneumonia_file_size.png",
            "Every RGB-encoded image belongs to Viral Pneumonia; the other classes are grayscale.",
            width="stretch",
        )
    with intensity:
        image(
            LATEX_FIGURES_DIR / "brightness_contrast_boxplots.png",
            "Brightness and contrast differ significantly by class, although their distributions overlap.",
            width="stretch",
        )
    source("Select a tab to keep one audit finding in focus.")


def pipeline():
    metadata_path = DATA_DIR / "arrays" / "preprocessing.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        metadata = {}
    samples = metadata.get("samples", {})
    evidence_header(
        "04 / The controls",
        "Make every experiment inherit the same rules.",
        "Preprocessing acts as experimental control: clean once, split once and record enough metadata to rebuild the result.",
    )
    _metric_cards(
        [
            ("Training", f"{samples.get('train', 14774):,}", "70%"),
            ("Validation", f"{samples.get('val', 3166):,}", "15%"),
            ("Test", f"{samples.get('test', 3166):,}", "15%"),
            ("Seed", str(metadata.get("preprocess_config", {}).get("random_state", 42)), "fixed"),
        ]
    )
    image(
        LATEX_FIGURES_DIR / "preprocessing_transform_steps.png",
        "Deduplicate → stratify → grayscale → align masks → normalize → augment training only.",
        width="stretch",
    )
    source("Split values are loaded from data/arrays/preprocessing.json.")


def baselines():
    entries = [
        ("Majority-class dummy", BASELINE_DIR / "dummy_metrics.json"),
        ("Logistic regression", BASELINE_DIR / "logistic_regression_metrics.json"),
        ("Histogram gradient boosting", BASELINE_DIR / "hist_gradient_boosting_metrics.json"),
    ]
    frame = comparison(entries)
    top_accuracy = frame["Accuracy"].max() if not frame.empty else None
    evidence_header(
        "05 / The floor",
        "Simple pixels were already surprisingly predictive.",
        "Deliberately plain models measured how much separability existed before deep learning entered the story.",
    )
    trio(
        "A neural model means little without an honest lower bound.",
        "Compare a majority guess, balanced logistic regression and histogram gradient boosting.",
        f"The strongest raw-pixel baseline reached {percent(top_accuracy)} test accuracy.",
    )
    metric_chart(frame, height=510)
    source()


def cnn():
    entries = [
        ("LeNet-5", CNN_DIR / "lenet_metrics.json"),
        ("Simple CNN", CNN_DIR / "cnn_simple_metrics.json"),
        ("Simple CNN · lung ROI", CNN_DIR / "cnn_simple_roi_baseline_lung_roi_metrics.json"),
        ("Scratch CNN", CNN_DIR / "cnn_scratch_metrics.json"),
    ]
    frame = comparison(entries)
    full = summary(CNN_DIR / "cnn_simple_metrics.json")
    roi = summary(CNN_DIR / "cnn_simple_roi_baseline_lung_roi_metrics.json")
    evidence_header(
        "06 / From scratch",
        "More capacity won. Lung restriction did not.",
        "Training from scratch made architecture and visible image region directly comparable without borrowed ImageNet features.",
    )
    chart_tab, matrix_tab = st.tabs(["Model comparison", "Best model errors"])
    with chart_tab:
        metric_chart(frame, height=520)
    with matrix_tab:
        image(
            CNN_DIR / "cnn_scratch_test_confusion_matrix.png",
            "Held-out confusion matrix for the strongest saved scratch CNN.",
            width="stretch",
        )
    callout(
        "The clue",
        f"Constraining the simple CNN to a lung ROI changed accuracy from {percent(full['accuracy'])} to "
        f"{percent(roi['accuracy'])} ({points((roi['accuracy'] or 0) - (full['accuracy'] or 0))}).",
    )
    source()


def transfer():
    entries = [
        ("Frozen EfficientNetB0", FULL),
        ("Flip + rotation", TRANSFER_DIR / "transfer_efficientnetb0_augmented_metrics.json"),
        ("Rotation only", TRANSFER_DIR / "transfer_efficientnetb0_augmented_no_flip_metrics.json"),
        ("Class weighted", TRANSFER_DIR / "transfer_efficientnetb0_class_weighted_metrics.json"),
        ("Fine-tuned", TRANSFER_DIR / "transfer_efficientnetb0_finetuned_metrics.json"),
    ]
    frame = comparison(entries)
    fine = summary(TRANSFER_DIR / "transfer_efficientnetb0_finetuned_metrics.json")
    evidence_header(
        "07 / Borrowed vision",
        "ImageNet knowledge transferred. Domain rules still mattered.",
        "EfficientNetB0 supplied mature visual features; controlled ablations tested augmentation, weighting and fine-tuning.",
    )
    comparison_tab, history_tab = st.tabs(["Experiment comparison", "Fine-tuning history"])
    with comparison_tab:
        metric_chart(frame, height=520)
    with history_tab:
        image(
            TRANSFER_DIR / "transfer_efficientnetb0_finetuned_history.png",
            "Training and validation history for the fine-tuned model.",
            width="stretch",
        )
    callout(
        "Domain lesson",
        f"Fine-tuning reached {percent(fine['accuracy'], 2)}. Horizontal mirroring hurt the corresponding augmented run.",
    )
    source()


def suspicion():
    evidence_header(
        "08 / The suspicion",
        "The heatmaps kept looking beyond the lungs.",
        "Grad-CAM served as a lead generator: useful for forming a hypothesis, insufficient for proving it.",
    )
    heatmaps, focus = st.tabs(["Where the model looked", "Lung focus versus chance"])
    with heatmaps:
        image(
            TRANSFER_DIR / "transfer_efficientnetb0_augmented_gradcam.png",
            "Representative Grad-CAM overlays. Green outlines mark lung masks.",
            width="stretch",
        )
    with focus:
        image(
            TRANSFER_DIR / "transfer_efficientnetb0_augmented_lung_focus.png",
            "Attention inside the lungs compared with the fraction expected from lung area alone.",
            width="stretch",
        )
    callout(
        "Correlation is not causation",
        "A coarse heatmap can spill across boundaries. The next experiment removes the suspected evidence from the input itself.",
    )


def causal_test():
    full = summary(FULL)
    masked = summary(MASKED)
    full_recall = class_score(FULL, "COVID")
    masked_recall = class_score(MASKED, "COVID")
    recall_delta = None if full_recall is None or masked_recall is None else masked_recall - full_recall
    evidence_header(
        "09 / The causal test",
        "Take away the background. Measure what breaks.",
        "The same EfficientNetB0 setup was retrained with every non-lung pixel forced to zero.",
    )
    _metric_cards(
        [
            ("Full-image accuracy", percent(full["accuracy"], 2), "test split"),
            ("Lungs-only accuracy", percent(masked["accuracy"], 2), points((masked["accuracy"] or 0) - (full["accuracy"] or 0))),
            ("Full-image COVID recall", percent(full_recall), "test split"),
            ("Lungs-only COVID recall", percent(masked_recall), points(recall_delta)),
        ]
    )
    recall_rows = [
        {
            "Class": name.replace("_", " "),
            "Full image": class_score(FULL, name),
            "Lungs only": class_score(MASKED, name),
        }
        for name in CLASSES
    ]
    metric_chart(pd.DataFrame(recall_rows), category="Class", height=440)
    verdict("What the intervention establishes", "Part of the COVID score depended on evidence outside the lungs.")
    source("Values are calculated from the current full-image and masked metrics files.")


def decomposition():
    entries = [("Full image", FULL), ("Lungs + texture", MASKED), ("Lung shape only", MASK_ONLY)]
    frame = comparison(entries)
    evidence_header(
        "10 / What survives",
        "Remove context. Then remove texture.",
        "A shape-only model isolates lung geometry from both background context and radiographic texture.",
    )
    comparison_tab, matrix_tab = st.tabs(["Signal decomposition", "Shape-only errors"])
    with comparison_tab:
        metric_chart(frame, height=500)
    with matrix_tab:
        image(
            TRANSFER_DIR / "transfer_efficientnetb0_mask_only_test_confusion_matrix.png",
            "Shape-only confusion matrix on the held-out split.",
            width="stretch",
        )
    covid_values = " · ".join(
        f"{label}: {percent(class_score(path, 'COVID'))}" for label, path in entries
    )
    callout("COVID recall", covid_values)
    verdict("Reading the decomposition", "Geometry carries signal. Texture adds signal. Background also carried label information.")
    source()


def limits():
    header(
        "11 / The boundary",
        "A strong portfolio model is still not a clinical model.",
        "The shortcut result predicts a specific failure mode: COVID recall may collapse when source and acquisition cues change.",
    )
    trio(
        "A stitched public collection cannot represent hospitals, scanners, subgroups or future prevalence.",
        "Translate every limitation into a validation requirement and a concrete next experiment.",
        "External, source-aware validation is mandatory; uncertainty and subgroup behavior remain unmeasured.",
    )
    left, right = st.columns(2, gap="large", vertical_alignment="top")
    with left:
        with st.container(border=True, height="stretch"):
            st.subheader("Before any clinical claim")
            st.markdown(
                "- Validate on multiple unseen hospitals and scanners\n"
                "- Split by acquisition source and patient identity\n"
                "- Report calibration and performance by subgroup\n"
                "- Define abstention and out-of-distribution behavior\n"
                "- Monitor the COVID-recall failure found here"
            )
    with right:
        with st.container(border=True, height="stretch"):
            st.subheader("What we would build next")
            st.markdown(
                "- Provenance-aware training and evaluation\n"
                "- Background randomization as a confound control\n"
                "- Full-image and lungs-only results side by side\n"
                "- External validation before more tuning\n"
                "- Error review with radiology expertise"
            )
    disclaimer()
    callout(
        "The honest claim",
        "This project classifies this dataset well and demonstrates why that fact alone does not establish trust elsewhere.",
    )


def verdict_page():
    best = summary(CNN_DIR / "cnn_scratch_metrics.json")
    full_recall = class_score(FULL, "COVID")
    masked_recall = class_score(MASKED, "COVID")
    header(
        "12 / The verdict",
        "The score was the beginning of the investigation.",
        "The strongest outcome is a reproducible chain of evidence from dataset audit to a causal model intervention.",
    )
    verdict("Headline", f"{percent(best['accuracy'], 2)} test accuracy — with a documented shortcut risk.")
    numbers(
        [
            (percent(best["accuracy"], 2), "best saved accuracy"),
            (percent(full_recall), "full-image COVID recall"),
            (percent(masked_recall), "lungs-only COVID recall"),
            (points((masked_recall or 0) - (full_recall or 0)), "recall change"),
        ]
    )
    st.markdown(
        '<div class="thesis">A trustworthy workflow tests <em>what the model uses</em>, '
        "not only whether its prediction matches a label.</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "The presentation ends here. Open **Live classifier** in the sidebar to run a saved model.",
        icon=":material/arrow_forward:",
    )
    disclaimer()
    source()


@st.cache_resource(show_spinner="Loading the saved model…", max_entries=6)
def _cached_model(path: str):
    return load_model(path)


def live_classifier():
    header(
        "13 / Working app",
        "Run the model. Keep the caveat in frame.",
        "Upload a chest X-ray, choose a saved classifier and inspect its probabilities and Grad-CAM overlay.",
        mode="app",
    )
    disclaimer()
    models = available_models()
    if not models:
        st.error("No compatible saved models were found under models/.")
        return
    labels = {model.label: model for model in models}
    controls, metric = st.columns([2, 1], gap="large", vertical_alignment="bottom")
    with controls:
        selected_label = st.selectbox("Saved classifier", list(labels), key="demo_model")
    selected = labels[selected_label]
    model_metrics = summary(selected.metrics)
    with metric:
        st.metric("Held-out test accuracy", percent(model_metrics["accuracy"], 2), selected.key)
    uploaded = st.file_uploader(
        "Chest X-ray",
        type=["png", "jpg", "jpeg"],
        help="The image is processed in memory and resized for the selected model.",
        key="demo_upload",
    )
    if uploaded is None:
        st.info(
            "Choose a PNG or JPEG chest X-ray to run inference. Out-of-distribution images may behave unpredictably.",
            icon=":material/upload_file:",
        )
        return
    raw = read_image(uploaded)
    tensor = prepare(raw, selected)
    result_slot = st.container()
    with result_slot.skeleton(height=420):
        model = _cached_model(str(selected.path))
        probabilities = predict(model, tensor)
        gradcam = explain(model, tensor)
    predicted_index = int(gradcam["predicted_index"])
    predicted_label = str(gradcam["predicted_label"])
    confidence = float(probabilities[predicted_index])
    verdict("Model output", f"{predicted_label.replace('_', ' ')} · {percent(confidence)}")
    input_tab, explanation_tab, scores_tab = st.tabs(["Input image", "Grad-CAM", "Class probabilities"])
    with input_tab:
        st.image(raw, width="stretch")
    with explanation_tab:
        st.image(gradcam["overlay"], width="stretch")
        st.caption("This is a coarse sensitivity map, not lesion localization or a clinical explanation.")
    rows = pd.DataFrame(
        {
            "Class": [name.replace("_", " ") for name in CLASSES],
            "Probability": [float(value) for value in probabilities],
        }
    )
    with scores_tab:
        probability_chart(rows)
        table = rows.copy()
        table["Probability"] = table["Probability"].map(percent)
        st.dataframe(table, hide_index=True, width="stretch")
    callout(
        "Read this output carefully",
        "Confidence is a softmax score on one image. It does not measure diagnostic certainty, dataset shift or whether the model used the right evidence.",
    )
    source(f"Inference ran locally with {selected.path.name}; metrics loaded from {selected.metrics.name}.")
