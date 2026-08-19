from __future__ import annotations

from tensorflow import keras

from ..config import CLASS_NAMES
from .config import TransferConfig

BACKBONE_BUILDERS = {
    "efficientnetb0": keras.applications.EfficientNetB0,
}


def build_transfer_model(config: TransferConfig = TransferConfig()) -> keras.Model:
    """Build an EfficientNet with a new classification head on top.

    Inputs are expected as float32 pixels in [0, 255]: EfficientNet's Keras
    implementation includes its own internal rescaling layer, so the images
    must NOT be pre-normalized to [0, 1] the way the sklearn baseline is.
    """
    backbone_builder = BACKBONE_BUILDERS[config.backbone]
    backbone = backbone_builder(
        include_top=False,
        weights="imagenet" if config.pretrained else None,
        input_shape=(*config.image_size, 3),
        pooling="avg",
    )
    backbone.trainable = not config.freeze_backbone

    inputs = keras.Input(shape=(*config.image_size, 3))
    features = backbone(inputs, training=False)
    x = keras.layers.Dropout(config.dropout_rate)(features)
    x = keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = keras.layers.Dropout(config.dropout_rate)(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"{config.backbone}_transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
