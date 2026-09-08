from __future__ import annotations

from dataclasses import replace

import tensorflow as tf
from tensorflow import keras

from ..config import CLASS_NAMES
from .config import TransferConfig

BACKBONE_BUILDERS = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetb4": keras.applications.EfficientNetB4,
    "resnet50": keras.applications.ResNet50,
}

BACKBONE_PREPROCESSORS = {
    "efficientnetb0": None,
    "efficientnetb4": None,
    "resnet50": keras.applications.resnet50.preprocess_input,
}

IMAGE_INPUT_NAME = "image"
MASK_INPUT_NAME = "mask"

TRANSFER_CUSTOM_OBJECTS = {}


@keras.utils.register_keras_serializable(package="covid_xray")
class MaskedGlobalAveragePooling2D(keras.layers.Layer):
    def __init__(self, mask_threshold: int = 127, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_threshold = mask_threshold

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        features, mask = inputs
        spatial = tf.shape(features)[1:3]
        mask_resized = tf.image.resize(mask, spatial, method="nearest")
        mask_binary = tf.cast(mask_resized > self.mask_threshold, features.dtype)
        weighted = features * mask_binary
        sum_features = tf.reduce_sum(weighted, axis=[1, 2])
        count = tf.reduce_sum(mask_binary, axis=[1, 2]) + tf.keras.backend.epsilon()
        return sum_features / count

    def get_config(self) -> dict:
        config = super().get_config()
        config["mask_threshold"] = self.mask_threshold
        return config


TRANSFER_CUSTOM_OBJECTS["MaskedGlobalAveragePooling2D"] = MaskedGlobalAveragePooling2D


def build_transfer_model(
    config: TransferConfig = TransferConfig(),
    *,
    backbone_training: bool | None = None,
) -> keras.Model:
    """Build a pretrained backbone with a new classification head on top.

    Inputs are expected as float32 pixels in [0, 255]. EfficientNet's Keras
    implementation includes its own internal rescaling/normalization layers,
    so its images must NOT be pre-normalized to [0, 1] the way the sklearn
    baseline is. ResNet50 has no such built-in layer, so its backbone-specific
    `preprocess_input` (BGR channel reorder + ImageNet mean subtraction) is
    applied automatically as part of the model graph via `BACKBONE_PREPROCESSORS`.

    When ``config.masked_pooling`` is enabled the model takes two inputs
    (image and lung mask) and applies masked global average pooling over the
    backbone's spatial feature maps instead of averaging the full feature map.
    """
    backbone_builder = BACKBONE_BUILDERS[config.backbone]
    pooling = None if config.masked_pooling else "avg"
    backbone = backbone_builder(
        include_top=False,
        weights="imagenet" if config.pretrained else None,
        input_shape=(*config.image_size, 3),
        pooling=pooling,
    )
    backbone.trainable = not config.freeze_backbone
    if backbone_training is None:
        backbone_training = not config.freeze_backbone

    image_input = keras.Input(shape=(*config.image_size, 3), name=IMAGE_INPUT_NAME)
    preprocess = BACKBONE_PREPROCESSORS.get(config.backbone)
    backbone_inputs = (
        keras.layers.Lambda(preprocess, name="preprocess_input")(image_input)
        if preprocess is not None
        else image_input
    )
    features = backbone(backbone_inputs, training=backbone_training)

    if config.masked_pooling:
        mask_input = keras.Input(shape=(*config.image_size, 1), name=MASK_INPUT_NAME)
        pooled = MaskedGlobalAveragePooling2D(mask_threshold=config.mask_threshold)(
            (features, mask_input)
        )
        model_inputs = [image_input, mask_input]
    else:
        pooled = features
        model_inputs = image_input

    x = keras.layers.Dropout(config.dropout_rate)(pooled)
    x = keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = keras.layers.Dropout(config.dropout_rate)(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(model_inputs, outputs, name=f"{config.backbone}_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def freeze_all_but_last_layers(backbone: keras.Model, unfreeze_layers: int) -> None:
    backbone.trainable = True
    if unfreeze_layers <= 0:
        return
    cutoff = max(len(backbone.layers) - unfreeze_layers, 0)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False


def prepare_for_fine_tuning(
    model: keras.Model, config: TransferConfig = TransferConfig()
) -> keras.Model:
    weights = model.get_weights()
    fine_tune_config = replace(config, freeze_backbone=False, fine_tune=False)
    fine_tuned_model = build_transfer_model(
        fine_tune_config, backbone_training=True
    )
    fine_tuned_model.set_weights(weights)
    if config.fine_tune_unfreeze_layers > 0:
        backbone = fine_tuned_model.get_layer(config.backbone)
        freeze_all_but_last_layers(backbone, config.fine_tune_unfreeze_layers)
    fine_tuned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return fine_tuned_model
