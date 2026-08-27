from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from covid_xray.cnn import CNNConfig, build_lenet_model, build_model
from covid_xray.cnn.config import DEFAULT_LENET_IMAGE_SIZE, default_image_size
from covid_xray.config import CLASS_NAMES

LENET = CNNConfig(architecture="lenet", image_size=DEFAULT_LENET_IMAGE_SIZE)


def test_output_shape_matches_num_classes() -> None:
    model = build_lenet_model(LENET)

    assert model.output_shape == (None, len(CLASS_NAMES))


def test_native_resolution_stays_a_low_capacity_floor() -> None:
    model = build_lenet_model(LENET)

    # ~61k parameters: an order of magnitude below the scratch model, which is
    # the entire point of including it.
    assert model.count_params() < 100_000


def test_is_much_smaller_than_the_scratch_model() -> None:
    lenet = build_model(LENET)
    scratch = build_model(CNNConfig(architecture="scratch"))

    assert lenet.count_params() * 10 < scratch.count_params()


def test_topology_follows_the_original_paper() -> None:
    model = build_lenet_model(LENET)
    names = [layer.name for layer in model.layers]

    assert names[-5:] == ["S4", "flatten", "C5", "F6", "predictions"]
    assert model.get_layer("C1").filters == 6
    assert model.get_layer("C3").filters == 16
    assert model.get_layer("C5").units == 120
    assert model.get_layer("F6").units == 84


def test_original_variant_uses_tanh_and_average_pooling() -> None:
    model = build_lenet_model(replace(LENET, lenet_variant="original"))

    assert model.get_layer("C1").activation.__name__ == "tanh"
    assert "Average" in type(model.get_layer("S2")).__name__


def test_modern_variant_uses_relu_and_max_pooling() -> None:
    model = build_lenet_model(replace(LENET, lenet_variant="modern"))

    assert model.get_layer("C1").activation.__name__ == "relu"
    assert "Max" in type(model.get_layer("S2")).__name__


def test_high_resolution_lenet_warns_about_its_dense_head() -> None:
    # LeNet pools only twice, so 224x224 leaves a 53x53x16 map and the dense
    # head balloons past the scratch model. Silently training that would
    # misrepresent it as a floor.
    with pytest.warns(RuntimeWarning, match="low-capacity"):
        build_lenet_model(replace(LENET, image_size=(224, 224)))


def test_predicts_a_probability_distribution() -> None:
    model = build_model(LENET)
    batch = np.random.default_rng(0).uniform(0, 255, size=(2, 32, 32, 1)).astype("float32")

    predictions = model.predict(batch, verbose=0)

    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_rescaling_is_inside_the_model() -> None:
    model = build_lenet_model(LENET)

    assert any(layer.name == "rescale" for layer in model.layers)


def test_default_image_size_differs_by_architecture() -> None:
    assert default_image_size("lenet") == (32, 32)
    assert default_image_size("scratch") != (32, 32)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="architecture"):
        CNNConfig(architecture="resnet")


def test_unknown_lenet_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="lenet_variant"):
        CNNConfig(lenet_variant="vintage")


def test_both_architectures_compile_identically() -> None:
    # A difference in the comparison table must come from the architecture,
    # not from a stray optimizer or loss difference.
    lenet = build_model(replace(LENET, learning_rate=1e-3))
    scratch = build_model(CNNConfig(architecture="scratch", learning_rate=1e-3))

    assert lenet.loss == scratch.loss
    assert type(lenet.optimizer) is type(scratch.optimizer)
