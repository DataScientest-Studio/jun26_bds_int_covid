from __future__ import annotations

import numpy as np

from cnn_helpers import SMALL_CONFIG

from covid_xray.cnn import CNNConfig, build_cnn_model
from covid_xray.config import CLASS_NAMES


def test_output_shape_matches_num_classes() -> None:
    model = build_cnn_model(SMALL_CONFIG)

    assert model.output_shape == (None, len(CLASS_NAMES))


def test_input_is_single_channel() -> None:
    model = build_cnn_model(SMALL_CONFIG)

    assert model.input_shape == (None, 16, 16, 1)


def test_predicts_a_probability_distribution() -> None:
    model = build_cnn_model(SMALL_CONFIG)
    batch = np.random.default_rng(0).uniform(0, 255, size=(2, 16, 16, 1)).astype("float32")

    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, len(CLASS_NAMES))
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_rescaling_is_inside_the_model() -> None:
    # The saved model must be self-contained: callers pass [0, 255] pixels and
    # the normalization travels with the artifact.
    model = build_cnn_model(SMALL_CONFIG)

    assert any(layer.name == "rescale" for layer in model.layers)


def test_default_configuration_stays_a_small_model() -> None:
    model = build_cnn_model(CNNConfig())

    assert model.count_params() < 2_000_000
