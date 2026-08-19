from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..config import (
    CLASS_COLUMN,
    CLASS_NAMES,
    ID_TO_LABEL,
    IMAGE_PATH_COLUMN,
    MASK_PATH_COLUMN,
    RANDOM_STATE,
)
from ..preprocessing.transforms import read_grayscale, resize_mask
from .config import TransferConfig
from .dataset import load_image

DEFAULT_ALPHA = 0.4
DEFAULT_BACKBONE_LAYER = "efficientnetb0"
DEFAULT_MASK_THRESHOLD = 127


def find_last_conv_layer_name(backbone: keras.Model) -> str:
    for layer in reversed(backbone.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError(f"No Conv2D layer found in {backbone.name!r}")


def build_gradcam_models(
    model: keras.Model,
    backbone_layer_name: str = DEFAULT_BACKBONE_LAYER,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[keras.Model, keras.Model]:
    backbone = model.get_layer(backbone_layer_name)
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(backbone)
    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    backbone_grad_model = keras.Model(backbone.inputs, [last_conv_layer.output, backbone.output])

    classifier_input = keras.Input(shape=backbone.output.shape[1:])
    x = classifier_input
    after_backbone = False
    for layer in model.layers:
        if layer.name == backbone_layer_name:
            after_backbone = True
            continue
        if after_backbone:
            x = layer(x)
    classifier_model = keras.Model(classifier_input, x)

    return backbone_grad_model, classifier_model


def compute_gradcam_heatmap(
    image_array: np.ndarray,
    backbone_grad_model: keras.Model,
    classifier_model: keras.Model,
    pred_index: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    inputs = tf.convert_to_tensor(image_array)
    if inputs.ndim == 3:
        inputs = inputs[tf.newaxis, ...]

    with tf.GradientTape() as tape:
        conv_output, pooled_features = backbone_grad_model(inputs)
        tape.watch(conv_output)
        predictions = classifier_model(pooled_features)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy(), pred_index


def resize_heatmap(heatmap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    resized = tf.image.resize(heatmap[..., np.newaxis], image_size)
    return resized.numpy()[..., 0]


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    from matplotlib import colormaps

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colors = colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8] * 255

    superimposed = jet_heatmap * alpha + image.astype("float32")
    return np.clip(superimposed, 0, 255).astype("uint8")


def load_image_for_gradcam(image_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    return load_image(tf.constant(image_path), image_size).numpy()


def load_mask_for_gradcam(mask_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    mask = read_grayscale(mask_path)
    return resize_mask(mask, image_size)


def lung_attention_fraction(
    heatmap: np.ndarray, mask: np.ndarray, threshold: int = DEFAULT_MASK_THRESHOLD
) -> float:
    """Share of the Grad-CAM heatmap's total activation that falls inside the lung mask.

    A value near the lung area's share of the image (its "chance" level) means the
    model attends to the lungs no more than a spatially uninformative model would;
    a much higher value means the model is genuinely keying off lung tissue rather
    than borders, artifacts, or other background content.
    """
    is_lung = mask > threshold
    total_energy = float(heatmap.sum())
    if total_energy <= 0:
        return float("nan")
    return float(heatmap[is_lung].sum() / total_energy)


def draw_mask_contour(
    image: np.ndarray,
    mask: np.ndarray,
    threshold: int = DEFAULT_MASK_THRESHOLD,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    import cv2

    binary_mask = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = image.copy()
    cv2.drawContours(outlined, contours, -1, color, 1)
    return outlined


def gradcam_for_image(
    model: keras.Model,
    image_array: np.ndarray,
    backbone_grad_model: keras.Model,
    classifier_model: keras.Model,
    mask: Optional[np.ndarray] = None,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
) -> dict:
    heatmap, pred_index = compute_gradcam_heatmap(image_array, backbone_grad_model, classifier_model)
    heatmap_resized = resize_heatmap(heatmap, image_array.shape[:2])
    overlay = overlay_heatmap(image_array.astype("uint8"), heatmap_resized)

    result = {
        "image": image_array.astype("uint8"),
        "heatmap": heatmap_resized,
        "overlay": overlay,
        "predicted_label": ID_TO_LABEL[pred_index],
        "lung_fraction": None,
    }
    if mask is not None:
        result["mask"] = mask
        result["lung_fraction"] = lung_attention_fraction(heatmap_resized, mask, mask_threshold)
        result["overlay"] = draw_mask_contour(overlay, mask, mask_threshold)
    return result


def save_gradcam_grid(
    model: keras.Model,
    frame: pd.DataFrame,
    output_path: Path | str,
    image_size: Tuple[int, int] = TransferConfig().image_size,
    backbone_layer_name: str = DEFAULT_BACKBONE_LAYER,
    last_conv_layer_name: Optional[str] = None,
    samples_per_class: int = 1,
    random_state: int = RANDOM_STATE,
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backbone_grad_model, classifier_model = build_gradcam_models(
        model, backbone_layer_name, last_conv_layer_name
    )

    samples = pd.concat(
        [
            group.sample(min(samples_per_class, len(group)), random_state=random_state)
            for _, group in frame.groupby(CLASS_COLUMN)
        ]
    ).reset_index(drop=True)

    n_rows = len(samples)
    figure = Figure(figsize=(9, 3 * n_rows))
    FigureCanvasAgg(figure)

    has_masks = MASK_PATH_COLUMN in samples.columns

    for row_index, row in samples.iterrows():
        image_array = load_image_for_gradcam(row[IMAGE_PATH_COLUMN], image_size)
        mask = None
        if has_masks and Path(row[MASK_PATH_COLUMN]).exists():
            mask = load_mask_for_gradcam(row[MASK_PATH_COLUMN], image_size)
        result = gradcam_for_image(
            model, image_array, backbone_grad_model, classifier_model, mask=mask
        )

        axes = [figure.add_subplot(n_rows, 3, row_index * 3 + i + 1) for i in range(3)]
        axes[0].imshow(result["image"])
        axes[0].set_title(f"{row[CLASS_COLUMN]} (true)")
        axes[1].imshow(result["heatmap"], cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(result["overlay"])
        title = f"predicted: {result['predicted_label']}"
        if result["lung_fraction"] is not None:
            title += f"\nlung focus: {result['lung_fraction']:.0%}"
        axes[2].set_title(title)
        for axis in axes:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def summarize_lung_focus(
    model: keras.Model,
    frame: pd.DataFrame,
    image_size: Tuple[int, int] = TransferConfig().image_size,
    backbone_layer_name: str = DEFAULT_BACKBONE_LAYER,
    last_conv_layer_name: Optional[str] = None,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    sample_size: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute, per image, how much Grad-CAM attention falls inside the lung mask.

    Returns one row per image with the true/predicted label, the observed lung
    attention fraction, and the mask's own pixel coverage (the "chance" fraction
    a spatially uninformative model would get by construction).
    """
    if MASK_PATH_COLUMN not in frame.columns:
        raise ValueError(f"frame needs a {MASK_PATH_COLUMN!r} column to compute lung focus")

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(sample_size, random_state=random_state)

    backbone_grad_model, classifier_model = build_gradcam_models(
        model, backbone_layer_name, last_conv_layer_name
    )

    records = []
    for _, row in frame.iterrows():
        image_array = load_image_for_gradcam(row[IMAGE_PATH_COLUMN], image_size)
        heatmap, pred_index = compute_gradcam_heatmap(image_array, backbone_grad_model, classifier_model)
        heatmap_resized = resize_heatmap(heatmap, image_size)
        mask = load_mask_for_gradcam(row[MASK_PATH_COLUMN], image_size)
        is_lung = mask > mask_threshold

        records.append(
            {
                "true_label": row[CLASS_COLUMN],
                "predicted_label": ID_TO_LABEL[pred_index],
                "lung_fraction": lung_attention_fraction(heatmap_resized, mask, mask_threshold),
                "mask_coverage": float(is_lung.mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def aggregate_lung_focus(summary: pd.DataFrame) -> pd.DataFrame:
    per_class = summary.groupby("true_label").agg(
        mean_lung_fraction=("lung_fraction", "mean"),
        std_lung_fraction=("lung_fraction", "std"),
        mean_mask_coverage=("mask_coverage", "mean"),
        n_images=("lung_fraction", "count"),
    )
    per_class["lung_focus_vs_chance"] = (
        per_class["mean_lung_fraction"] - per_class["mean_mask_coverage"]
    )
    return per_class.reindex([name for name in CLASS_NAMES if name in per_class.index])


def aggregate_lung_focus_by_correctness(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare Grad-CAM lung focus between correct and misclassified predictions.

    If the model attends to the lungs noticeably less on the images it gets
    wrong, that is evidence its errors coincide with it leaning on background
    or border cues rather than the pathology itself.
    """
    frame = summary.copy()
    frame["prediction"] = np.where(
        frame["true_label"] == frame["predicted_label"], "correct", "misclassified"
    )
    per_group = frame.groupby("prediction").agg(
        mean_lung_fraction=("lung_fraction", "mean"),
        std_lung_fraction=("lung_fraction", "std"),
        mean_mask_coverage=("mask_coverage", "mean"),
        n_images=("lung_fraction", "count"),
    )
    per_group["lung_focus_vs_chance"] = (
        per_group["mean_lung_fraction"] - per_group["mean_mask_coverage"]
    )
    return per_group.reindex(["correct", "misclassified"]).dropna(how="all")


def _plot_lung_focus_bars(per_group: pd.DataFrame, title: str, output_path: Path) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    positions = np.arange(len(per_group))
    axis.bar(positions - 0.2, per_group["mean_lung_fraction"], width=0.4, label="Grad-CAM in lungs")
    axis.bar(positions + 0.2, per_group["mean_mask_coverage"], width=0.4, label="lung area (chance)")
    axis.set_xticks(positions)
    axis.set_xticklabels(per_group.index, rotation=45, ha="right")
    axis.set_ylabel("Fraction")
    axis.set_ylim(0, 1)
    axis.set_title(title, fontsize=10)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def save_lung_focus_report(
    summary: pd.DataFrame, output_dir: Path | str, model_name: str
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class = aggregate_lung_focus(summary)

    csv_path = output_dir / f"{model_name}_lung_focus.csv"
    per_class.to_csv(csv_path)

    png_path = output_dir / f"{model_name}_lung_focus.png"
    _plot_lung_focus_bars(
        per_class, f"{model_name}\nGrad-CAM attention inside lungs vs. chance", png_path
    )

    return csv_path, png_path


def save_correctness_lung_focus_report(
    summary: pd.DataFrame, output_dir: Path | str, model_name: str
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_group = aggregate_lung_focus_by_correctness(summary)

    csv_path = output_dir / f"{model_name}_lung_focus_by_correctness.csv"
    per_group.to_csv(csv_path)

    png_path = output_dir / f"{model_name}_lung_focus_by_correctness.png"
    _plot_lung_focus_bars(
        per_group, f"{model_name}\nGrad-CAM lung focus: correct vs misclassified", png_path
    )

    return csv_path, png_path
