from __future__ import annotations

import pytest

from covid_xray.config import CLASS_NAMES
from covid_xray.transfer_learning import TransferConfig, build_transfer_model, prepare_for_fine_tuning

SMALL = TransferConfig(image_size=(64, 64), pretrained=False, dense_units=8)


def test_build_transfer_model_output_shape_matches_num_classes() -> None:
    model = build_transfer_model(SMALL)

    assert model.output_shape == (None, len(CLASS_NAMES))


def test_build_transfer_model_freezes_backbone_by_default() -> None:
    model = build_transfer_model(SMALL)
    backbone = next(layer for layer in model.layers if layer.name.startswith("efficientnet"))

    assert backbone.trainable is False


def test_build_transfer_model_can_unfreeze_backbone() -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, dense_units=8, freeze_backbone=False
    )
    model = build_transfer_model(config)
    backbone = next(layer for layer in model.layers if layer.name.startswith("efficientnet"))

    assert backbone.trainable is True


def test_build_transfer_model_predicts_a_probability_distribution() -> None:
    import numpy as np

    model = build_transfer_model(SMALL)
    batch = np.random.default_rng(0).uniform(0, 255, size=(2, 64, 64, 3)).astype("float32")

    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, len(CLASS_NAMES))
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_prepare_for_fine_tuning_unfreezes_backbone_and_lowers_learning_rate() -> None:
    model = build_transfer_model(SMALL)
    fine_tuned = prepare_for_fine_tuning(
        model,
        TransferConfig(
            image_size=(64, 64),
            pretrained=False,
            dense_units=8,
            fine_tune_learning_rate=1e-5,
        ),
    )
    backbone = next(
        layer for layer in fine_tuned.layers if layer.name.startswith("efficientnet")
    )

    assert backbone.trainable is True
    assert fine_tuned.optimizer.learning_rate.numpy() == pytest.approx(1e-5)


def test_transfer_config_rejects_fine_tune_without_pretrained() -> None:
    with pytest.raises(ValueError, match="pretrained"):
        TransferConfig(fine_tune=True, pretrained=False)
