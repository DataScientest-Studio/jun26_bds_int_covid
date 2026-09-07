from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..config import ID_TO_LABEL


DEFAULT_ALPHA = 0.4
DEFAULT_MASK_THRESHOLD = 127


def find_gradcam_target(
    model: keras.Model,
    target_layer_name: Optional[str] = None,
) -> tuple[keras.Model, keras.layers.Layer]:
    """
    Find the convolutional layer used for Grad-CAM.

    Returns
    -------
    container_model:
        Either the full model itself or a nested backbone model.

    target_layer:
        The Conv2D layer whose feature maps will be used.
    """

    # --------------------------------------------------
    # Explicit layer name
    # --------------------------------------------------
    if target_layer_name is not None:

        # First check top-level layers
        for layer in model.layers:
            if (
                layer.name == target_layer_name
                and isinstance(layer, keras.layers.Conv2D)
            ):
                return model, layer

        # Then check one level of nested models
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                try:
                    target = layer.get_layer(target_layer_name)
                except ValueError:
                    continue

                if isinstance(target, keras.layers.Conv2D):
                    return layer, target

        raise ValueError(
            f"Could not find Conv2D layer "
            f"{target_layer_name!r} in model {model.name!r}"
        )

    # --------------------------------------------------
    # Standard CNN:
    # search top-level Conv2D layers first
    # --------------------------------------------------
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return model, layer

    # --------------------------------------------------
    # Transfer learning:
    # search inside nested backbone
    # --------------------------------------------------
    for layer in reversed(model.layers):

        if not isinstance(layer, keras.Model):
            continue

        for inner_layer in reversed(layer.layers):
            if isinstance(
                inner_layer,
                keras.layers.Conv2D,
            ):
                return layer, inner_layer

    raise ValueError(
        f"No Conv2D layer found in model {model.name!r}"
    )

def build_gradcam_models(
    model: keras.Model,
    target_layer_name: Optional[str] = None,
) -> tuple[keras.Model, Optional[keras.Model]]:
    """
    Build models needed for Grad-CAM.

    For ordinary CNNs:
        grad_model directly outputs
        [conv_features, predictions].

    For nested transfer-learning models:
        one model extracts backbone conv features,
        another represents the classifier head.
    """

    container, target_layer = find_gradcam_target(
        model,
        target_layer_name,
    )

    # ==================================================
    # Case 1:
    # Conv layer belongs directly to the model
    #
    # Simple CNN
    # Scratch CNN
    # LeNet
    # ==================================================
    if container is model:

        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                target_layer.output,
                model.output,
            ],
            name=f"{model.name}_gradcam",
        )

        return grad_model, None

    # ==================================================
    # Case 2:
    # Conv layer is inside a nested backbone
    #
    # e.g. EfficientNetB0
    # ==================================================

    backbone = container

    backbone_grad_model = keras.Model(
        inputs=backbone.inputs,
        outputs=[
            target_layer.output,
            backbone.output,
        ],
        name=f"{backbone.name}_gradcam",
    )

    # Rebuild everything AFTER the backbone.
    classifier_input = keras.Input(
        shape=backbone.output.shape[1:],
        dtype=backbone.output.dtype,
        name="gradcam_classifier_input",
    )

    x = classifier_input
    after_backbone = False

    for layer in model.layers:

        if layer is backbone:
            after_backbone = True
            continue

        if after_backbone:
            x = layer(x)

    classifier_model = keras.Model(
        classifier_input,
        x,
        name=f"{model.name}_classifier",
    )

    return backbone_grad_model, classifier_model

def compute_gradcam_heatmap(
    image_array: np.ndarray,
    grad_model: keras.Model,
    classifier_model: Optional[keras.Model] = None,
    pred_index: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM for one image.

    image_array can be:
        H x W x 1
        H x W x 3
    """

    inputs = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    if inputs.ndim == 3:
        inputs = inputs[tf.newaxis, ...]

    with tf.GradientTape() as tape:

        conv_output, features_or_predictions = grad_model(
            inputs,
            training=False,
        )

        tape.watch(conv_output)

        # Normal CNN
        if classifier_model is None:
            predictions = features_or_predictions

        # Transfer-learning model
        else:
            predictions = classifier_model(
                features_or_predictions,
                training=False,
            )

        if pred_index is None:
            pred_index = int(
                tf.argmax(predictions[0])
            )

        class_score = predictions[:, pred_index]

    gradients = tape.gradient(
        class_score,
        conv_output,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are None. "
            "The selected convolutional layer is not connected "
            "to the classifier output."
        )

    # Average gradient for each channel
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_output = conv_output[0]

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1,
    )

    # Grad-CAM normally keeps positive influence
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy(), pred_index

def resize_heatmap(
    heatmap: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:

    resized = tf.image.resize(
        heatmap[..., np.newaxis],
        image_size,
    )

    return resized.numpy()[..., 0]

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:

    from matplotlib import colormaps

    # Convert grayscale to RGB for visualization
    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape[-1] == 1:
        image = np.repeat(
            image,
            3,
            axis=-1,
        )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    jet_colors = colormaps["jet"](
        np.arange(256)
    )[:, :3]

    jet_heatmap = (
        jet_colors[heatmap_uint8]
        * 255
    )

    superimposed = (
        jet_heatmap * alpha
        + image.astype("float32")
    )

    return np.clip(
        superimposed,
        0,
        255,
    ).astype("uint8")

def lung_attention_fraction(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold: int = DEFAULT_MASK_THRESHOLD,
) -> float:

    is_lung = mask > threshold

    total_energy = float(
        heatmap.sum()
    )

    if total_energy <= 0:
        return float("nan")

    lung_energy = float(
        heatmap[is_lung].sum()
    )

    return lung_energy / total_energy

def draw_mask_contour(
    image: np.ndarray,
    mask: np.ndarray,
    threshold: int = DEFAULT_MASK_THRESHOLD,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:

    import cv2

    binary_mask = (
        mask > threshold
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    outlined = image.copy()

    cv2.drawContours(
        outlined,
        contours,
        -1,
        color,
        1,
    )

    return outlined

def gradcam_for_image(
    model: keras.Model,
    image_array: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_layer_name: Optional[str] = None,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
) -> dict:

    grad_model, classifier_model = build_gradcam_models(
        model,
        target_layer_name,
    )

    heatmap, pred_index = compute_gradcam_heatmap(
        image_array,
        grad_model,
        classifier_model,
    )

    heatmap = resize_heatmap(
        heatmap,
        image_array.shape[:2],
    )

    # display version
    display_image = image_array

    if display_image.shape[-1] == 1:
        display_image = np.repeat(
            display_image,
            3,
            axis=-1,
        )

    display_image = display_image.astype(
        "uint8"
    )

    overlay = overlay_heatmap(
        display_image,
        heatmap,
    )

    result = {
        "image": display_image,
        "heatmap": heatmap,
        "overlay": overlay,
        "predicted_index": pred_index,
        "predicted_label": ID_TO_LABEL[pred_index],
        "lung_fraction": None,
    }

    if mask is not None:

        result["mask"] = mask

        result["lung_fraction"] = (
            lung_attention_fraction(
                heatmap,
                mask,
                mask_threshold,
            )
        )

        result["overlay"] = draw_mask_contour(
            overlay,
            mask,
            mask_threshold,
        )

    return result

