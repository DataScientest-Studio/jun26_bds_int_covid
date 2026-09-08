from __future__ import annotations

import pytest

from covid_xray.config import CLASS_NAMES
from covid_xray.transfer_learning import TransferConfig, build_transfer_model, prepare_for_fine_tuning
from covid_xray.transfer_learning.model import (
    IMAGE_INPUT_NAME,
    MASK_INPUT_NAME,
    MaskedGlobalAveragePooling2D,
    freeze_all_but_last_layers,
)

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


EFFICIENTNETB4_SMALL = TransferConfig(
    image_size=(64, 64), backbone="efficientnetb4", pretrained=False, dense_units=8
)

RESNET_SMALL = TransferConfig(
    image_size=(64, 64), backbone="resnet50", pretrained=False, dense_units=8
)


def test_build_transfer_model_supports_efficientnetb4_backbone() -> None:
    model = build_transfer_model(EFFICIENTNETB4_SMALL)

    assert model.output_shape == (None, len(CLASS_NAMES))
    assert any(layer.name == "efficientnetb4" for layer in model.layers)


def test_build_transfer_model_freezes_efficientnetb4_backbone_by_default() -> None:
    model = build_transfer_model(EFFICIENTNETB4_SMALL)
    backbone = model.get_layer("efficientnetb4")

    assert backbone.trainable is False


def test_build_transfer_model_skips_preprocess_input_for_efficientnetb4() -> None:
    model = build_transfer_model(EFFICIENTNETB4_SMALL)

    assert not any(layer.name == "preprocess_input" for layer in model.layers)


def test_build_transfer_model_efficientnetb4_predicts_a_probability_distribution() -> None:
    import numpy as np

    model = build_transfer_model(EFFICIENTNETB4_SMALL)
    batch = np.random.default_rng(0).uniform(0, 255, size=(2, 64, 64, 3)).astype("float32")

    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, len(CLASS_NAMES))
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_build_transfer_model_supports_resnet50_backbone() -> None:
    model = build_transfer_model(RESNET_SMALL)

    assert model.output_shape == (None, len(CLASS_NAMES))
    assert any(layer.name == "resnet50" for layer in model.layers)


def test_build_transfer_model_freezes_resnet50_backbone_by_default() -> None:
    model = build_transfer_model(RESNET_SMALL)
    backbone = model.get_layer("resnet50")

    assert backbone.trainable is False


def test_build_transfer_model_applies_preprocess_input_for_resnet50() -> None:
    model = build_transfer_model(RESNET_SMALL)

    assert any(layer.name == "preprocess_input" for layer in model.layers)


def test_build_transfer_model_skips_preprocess_input_for_efficientnetb0() -> None:
    model = build_transfer_model(SMALL)

    assert not any(layer.name == "preprocess_input" for layer in model.layers)


def test_build_transfer_model_resnet50_predicts_a_probability_distribution() -> None:
    import numpy as np

    model = build_transfer_model(RESNET_SMALL)
    batch = np.random.default_rng(0).uniform(0, 255, size=(2, 64, 64, 3)).astype("float32")

    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, len(CLASS_NAMES))
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_transfer_config_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="backbone"):
        TransferConfig(backbone="resnet101")


def test_transfer_config_rejects_negative_fine_tune_unfreeze_layers() -> None:
    with pytest.raises(ValueError, match="fine_tune_unfreeze_layers"):
        TransferConfig(fine_tune_unfreeze_layers=-1)


def test_freeze_all_but_last_layers_keeps_only_tail_trainable() -> None:
    model = build_transfer_model(SMALL)
    backbone = model.get_layer("efficientnetb0")

    freeze_all_but_last_layers(backbone, unfreeze_layers=5)

    assert backbone.trainable is True
    trainable_flags = [layer.trainable for layer in backbone.layers]
    assert trainable_flags[-5:] == [True] * 5
    assert not any(trainable_flags[:-5])


def test_freeze_all_but_last_layers_zero_unfreezes_everything() -> None:
    model = build_transfer_model(SMALL)
    backbone = model.get_layer("efficientnetb0")

    freeze_all_but_last_layers(backbone, unfreeze_layers=0)

    assert backbone.trainable is True
    assert all(layer.trainable for layer in backbone.layers)


def test_prepare_for_fine_tuning_with_unfreeze_layers_freezes_early_layers() -> None:
    model = build_transfer_model(SMALL)
    fine_tuned = prepare_for_fine_tuning(
        model,
        TransferConfig(
            image_size=(64, 64),
            pretrained=False,
            dense_units=8,
            fine_tune_learning_rate=1e-5,
            fine_tune_unfreeze_layers=5,
        ),
    )
    backbone = fine_tuned.get_layer("efficientnetb0")

    trainable_flags = [layer.trainable for layer in backbone.layers]
    assert trainable_flags[-5:] == [True] * 5
    assert not any(trainable_flags[:-5])


def test_prepare_for_fine_tuning_default_unfreezes_all_backbone_layers() -> None:
    model = build_transfer_model(SMALL)
    fine_tuned = prepare_for_fine_tuning(
        model,
        TransferConfig(
            image_size=(64, 64), pretrained=False, dense_units=8, fine_tune_learning_rate=1e-5
        ),
    )
    backbone = fine_tuned.get_layer("efficientnetb0")

    assert all(layer.trainable for layer in backbone.layers)


def test_build_transfer_model_with_masked_pooling_has_two_inputs() -> None:
    config = TransferConfig(image_size=(64, 64), pretrained=False, dense_units=8, masked_pooling=True)
    model = build_transfer_model(config)

    assert isinstance(model.input, list)
    assert len(model.input) == 2
    assert model.get_layer(IMAGE_INPUT_NAME) is not None
    assert model.get_layer(MASK_INPUT_NAME) is not None
    assert model.output_shape == (None, len(CLASS_NAMES))


def test_build_transfer_model_with_masked_pooling_predicts_a_probability_distribution() -> None:
    import numpy as np

    config = TransferConfig(image_size=(64, 64), pretrained=False, dense_units=8, masked_pooling=True)
    model = build_transfer_model(config)
    images = np.random.default_rng(0).uniform(0, 255, size=(2, 64, 64, 3)).astype("float32")
    masks = np.zeros((2, 64, 64, 1), dtype="float32")
    masks[:, 16:48, 16:48, 0] = 255.0

    predictions = model.predict([images, masks], verbose=0)

    assert predictions.shape == (2, len(CLASS_NAMES))
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)


def test_masked_global_average_pooling_ignores_background_features() -> None:
    import tensorflow as tf

    layer = MaskedGlobalAveragePooling2D(mask_threshold=127)
    features = tf.constant([[[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]])
    mask = tf.constant([[[[0.0], [255.0]], [[255.0], [0.0]]]])

    pooled = layer((features, mask)).numpy()

    assert pooled.shape == (1, 2)
    assert pooled[0, 0] == pytest.approx(40.0)
    assert pooled[0, 1] == pytest.approx(50.0)


def test_transfer_config_rejects_masked_pooling_with_mask_only() -> None:
    with pytest.raises(ValueError, match="masked_pooling"):
        TransferConfig(masked_pooling=True, mask_only=True)
