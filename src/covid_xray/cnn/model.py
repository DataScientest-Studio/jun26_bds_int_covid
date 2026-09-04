from __future__ import annotations

import warnings

from tensorflow import keras

from ..config import CLASS_NAMES
from .config import LENET_PARAMETER_WARNING_THRESHOLD, CNNConfig

PIXEL_MAX_VALUE = 255.0


def build_scratch_model(config: CNNConfig = CNNConfig()) -> keras.Model:
    """Build a small VGG-style CNN from randomly initialised weights.

    Inputs are single-channel float32 pixels in [0, 255], the same range the
    transfer-learning datasets produce. Rescaling to [0, 1] happens inside the
    model via a Rescaling layer, so the saved `.keras` file is self-contained
    and cannot be fed wrongly-scaled images by a later caller.

    Each block is Conv-BN-ReLU twice, then max pooling. BatchNorm sits between
    the convolution and the activation, so the bias term is redundant
    (`use_bias=False`) and the layer is one tensor lighter.

    The head is GlobalAveragePooling rather than Flatten + Dense: on a 224x224
    input a Flatten would produce 50,176 features, and even a modest 256-unit
    Dense on top would add ~12.8M parameters -- which is where a from-scratch
    model on ~20k images overfits. GAP keeps the network at 1.18M parameters
    regardless of input resolution, and removes the model's ability to key on
    absolute pixel position.
    """
    inputs = keras.Input(shape=(*config.image_size, 1), dtype="float32", name="image")
    x = keras.layers.Rescaling(1.0 / PIXEL_MAX_VALUE, name="rescale")(inputs)

    for block_index, block_filters in enumerate(config.filters):
        for _ in range(2):
            x = keras.layers.Conv2D(
                block_filters,
                3,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Skip dropout in the first block: low-level edge filters are cheap and
        # generic, and there is nothing there worth regularizing away yet.
        if block_index > 0 and config.spatial_dropout_rate > 0:
            x = keras.layers.SpatialDropout2D(config.spatial_dropout_rate)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(config.dropout_rate)(x)
    outputs = keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", dtype="float32", name="predictions"
    )(x)

    return keras.Model(inputs, outputs, name="cnn_scratch")


def build_lenet_model(config: CNNConfig = CNNConfig()) -> keras.Model:
    """Build LeNet-5 (LeCun et al., 1998) as a low-capacity floor.

    Topology follows the original: 6 filters at 5x5 (valid padding), subsample
    by 2, 16 filters at 5x5, subsample by 2, then fully-connected 120 -> 84 ->
    n_classes.

    `lenet_variant` records which reading of "LeNet" this is:
      original -> tanh activations and average pooling, as published
      modern   -> ReLU and max pooling, as most reimplementations actually do

    Neither is more correct, but they are different models and the difference
    is worth stating rather than leaving to the reader to discover. The final
    RBF output layer of the 1998 paper is not reproduced; a softmax replaces
    it, as is now universal.

    Note on capacity: LeNet pools only twice, so the Flatten layer scales with
    the square of the input. At 32x32 the model holds ~61k parameters; at
    224x224 the fully-connected head alone is ~5.4M, which is more than the
    VGG-style model and defeats the purpose of running a floor at all. The
    warning below fires when that has happened.

    The Flatten + Dense head is also what makes this architecture the useful
    probe in the region experiment: unlike the GAP head, it can key directly
    on absolute pixel position. If the background shortcut in this dataset is
    positional, LeNet is the model best equipped to exploit it.
    """
    if config.lenet_variant == "original":
        activation = "tanh"
        pooling = keras.layers.AveragePooling2D
        initializer = "glorot_uniform"  # the sensible default for tanh
    else:
        activation = "relu"
        pooling = keras.layers.MaxPooling2D
        initializer = "he_normal"

    inputs = keras.Input(shape=(*config.image_size, 1), dtype="float32", name="image")
    x = keras.layers.Rescaling(1.0 / PIXEL_MAX_VALUE, name="rescale")(inputs)

    x = keras.layers.Conv2D(
        6, 5, padding="valid", activation=activation,
        kernel_initializer=initializer, name="C1",
    )(x)
    x = pooling(2, name="S2")(x)
    x = keras.layers.Conv2D(
        16, 5, padding="valid", activation=activation,
        kernel_initializer=initializer, name="C3",
    )(x)
    x = pooling(2, name="S4")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(
        120, activation=activation, kernel_initializer=initializer, name="C5"
    )(x)
    x = keras.layers.Dense(
        84, activation=activation, kernel_initializer=initializer, name="F6"
    )(x)
    outputs = keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name=f"lenet_{config.lenet_variant}")

    if model.count_params() > LENET_PARAMETER_WARNING_THRESHOLD:
        warnings.warn(
            f"LeNet at {config.image_size[0]}x{config.image_size[1]} has "
            f"{model.count_params():,} parameters, almost all in the "
            f"fully-connected head. LeNet pools only twice, so it is not a "
            f"low-capacity baseline at this resolution. Consider "
            f"image_size=(32, 32).",
            RuntimeWarning,
            stacklevel=2,
        )
    return model


BUILDERS = {
    "scratch": build_scratch_model,
    "lenet": build_lenet_model,
}


def build_model(config: CNNConfig = CNNConfig()) -> keras.Model:
    """Build and compile the architecture named by `config.architecture`.

    Compilation lives here rather than in the individual builders so every
    architecture is trained with an identical optimizer, loss, and metric.
    A difference in the comparison table then reflects the architecture and
    nothing else.
    """
    model = BUILDERS[config.architecture](config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Backwards-compatible alias: the scratch model was the only architecture when
# this package was written, and existing callers still ask for it by name.
def build_cnn_model(config: CNNConfig = CNNConfig()) -> keras.Model:
    return build_model(config)
