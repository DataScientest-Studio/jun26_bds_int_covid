from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covid_xray.transfer_learning import TransferConfig, build_transfer_model
from covid_xray.transfer_learning.gradcam import (
    aggregate_lung_focus,
    aggregate_lung_focus_by_correctness,
    build_gradcam_models,
    compute_gradcam_heatmap,
    draw_mask_contour,
    find_last_conv_layer_name,
    gradcam_for_image,
    load_mask_for_gradcam,
    lung_attention_fraction,
    overlay_heatmap,
    resize_heatmap,
    save_correctness_lung_focus_report,
    save_gradcam_grid,
    save_lung_focus_report,
    summarize_lung_focus,
)

SMALL = TransferConfig(image_size=(64, 64), pretrained=False, dense_units=8)


def test_find_last_conv_layer_name_returns_top_conv() -> None:
    model = build_transfer_model(SMALL)
    backbone = model.get_layer("efficientnetb0")

    assert find_last_conv_layer_name(backbone) == "top_conv"


def test_build_gradcam_models_wires_backbone_and_classifier_head() -> None:
    model = build_transfer_model(SMALL)

    backbone_grad_model, classifier_model = build_gradcam_models(model)

    assert classifier_model.output_shape == model.output_shape
    assert len(backbone_grad_model.outputs) == 2


def test_compute_gradcam_heatmap_has_expected_shape_and_range() -> None:
    model = build_transfer_model(SMALL)
    backbone_grad_model, classifier_model = build_gradcam_models(model)
    image = np.random.default_rng(0).uniform(0, 255, size=(64, 64, 3)).astype("float32")

    heatmap, pred_index = compute_gradcam_heatmap(image, backbone_grad_model, classifier_model)

    assert heatmap.shape == (2, 2)
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) <= 1.0
    assert 0 <= pred_index < model.output_shape[-1]


def test_resize_heatmap_matches_target_size() -> None:
    heatmap = np.random.default_rng(0).uniform(0, 1, size=(2, 2)).astype("float32")

    resized = resize_heatmap(heatmap, (64, 64))

    assert resized.shape == (64, 64)


def test_overlay_heatmap_returns_uint8_image_with_same_shape() -> None:
    image = np.random.default_rng(0).uniform(0, 255, size=(64, 64, 3)).astype("float32")
    heatmap = np.random.default_rng(0).uniform(0, 1, size=(64, 64)).astype("float32")

    overlay = overlay_heatmap(image.astype("uint8"), heatmap)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_gradcam_for_image_returns_expected_keys() -> None:
    model = build_transfer_model(SMALL)
    backbone_grad_model, classifier_model = build_gradcam_models(model)
    image = np.random.default_rng(0).uniform(0, 255, size=(64, 64, 3)).astype("float32")

    result = gradcam_for_image(model, image, backbone_grad_model, classifier_model)

    assert set(result) == {"image", "heatmap", "overlay", "predicted_label", "lung_fraction"}
    assert result["lung_fraction"] is None
    assert result["heatmap"].shape == (64, 64)
    assert result["overlay"].shape == (64, 64, 3)


def test_save_gradcam_grid_writes_a_png(manifest: pd.DataFrame, tmp_path: Path) -> None:
    model = build_transfer_model(SMALL)
    output_path = tmp_path / "gradcam.png"

    result = save_gradcam_grid(
        model, manifest, output_path, image_size=(64, 64), samples_per_class=1
    )

    assert result == output_path
    assert output_path.exists()


def test_lung_attention_fraction_is_one_when_heatmap_confined_to_lung() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1:3, 1:3] = 1.0

    assert lung_attention_fraction(heatmap, mask) == pytest.approx(1.0)


def test_lung_attention_fraction_is_zero_when_heatmap_confined_to_background() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[0, 0] = 1.0

    assert lung_attention_fraction(heatmap, mask) == pytest.approx(0.0)


def test_lung_attention_fraction_is_nan_for_empty_heatmap() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    heatmap = np.zeros((4, 4), dtype=np.float32)

    assert np.isnan(lung_attention_fraction(heatmap, mask))


def test_load_mask_for_gradcam_resizes_to_target_size(
    manifest_with_masks: pd.DataFrame,
) -> None:
    mask_path = manifest_with_masks["mask_path"].iloc[0]

    mask = load_mask_for_gradcam(mask_path, (64, 64))

    assert mask.shape == (64, 64)


def test_draw_mask_contour_keeps_image_shape() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255

    outlined = draw_mask_contour(image, mask)

    assert outlined.shape == image.shape
    assert outlined.dtype == np.uint8


def test_gradcam_for_image_reports_lung_fraction_when_mask_given() -> None:
    model = build_transfer_model(SMALL)
    backbone_grad_model, classifier_model = build_gradcam_models(model)
    image = np.random.default_rng(0).uniform(0, 255, size=(64, 64, 3)).astype("float32")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    result = gradcam_for_image(model, image, backbone_grad_model, classifier_model, mask=mask)

    assert result["lung_fraction"] is not None
    assert 0.0 <= result["lung_fraction"] <= 1.0


def test_summarize_lung_focus_returns_one_row_per_sampled_image(
    manifest_with_masks: pd.DataFrame,
) -> None:
    model = build_transfer_model(SMALL)

    summary = summarize_lung_focus(model, manifest_with_masks, image_size=(64, 64), sample_size=6)

    assert len(summary) == 6
    assert {"true_label", "predicted_label", "lung_fraction", "mask_coverage"} <= set(
        summary.columns
    )


def test_aggregate_lung_focus_has_one_row_per_class(
    manifest_with_masks: pd.DataFrame,
) -> None:
    model = build_transfer_model(SMALL)
    summary = summarize_lung_focus(model, manifest_with_masks, image_size=(64, 64))

    per_class = aggregate_lung_focus(summary)

    assert set(per_class.index) == set(manifest_with_masks["class"].unique())
    assert "lung_focus_vs_chance" in per_class.columns


def test_save_lung_focus_report_writes_csv_and_png(
    manifest_with_masks: pd.DataFrame, tmp_path: Path
) -> None:
    model = build_transfer_model(SMALL)
    summary = summarize_lung_focus(model, manifest_with_masks, image_size=(64, 64))

    csv_path, png_path = save_lung_focus_report(summary, tmp_path, "test_model")

    assert csv_path.exists()
    assert png_path.exists()


def test_aggregate_lung_focus_by_correctness_splits_correct_and_misclassified() -> None:
    summary = pd.DataFrame(
        {
            "true_label": ["COVID", "COVID", "Normal", "Normal"],
            "predicted_label": ["COVID", "Normal", "Normal", "COVID"],
            "lung_fraction": [0.5, 0.2, 0.6, 0.1],
            "mask_coverage": [0.25, 0.25, 0.25, 0.25],
        }
    )

    per_group = aggregate_lung_focus_by_correctness(summary)

    assert list(per_group.index) == ["correct", "misclassified"]
    assert per_group.loc["correct", "mean_lung_fraction"] == pytest.approx(0.55)
    assert per_group.loc["misclassified", "mean_lung_fraction"] == pytest.approx(0.15)
    assert per_group.loc["correct", "n_images"] == 2
    assert per_group.loc["misclassified", "n_images"] == 2


def test_aggregate_lung_focus_by_correctness_handles_all_correct() -> None:
    summary = pd.DataFrame(
        {
            "true_label": ["COVID", "Normal"],
            "predicted_label": ["COVID", "Normal"],
            "lung_fraction": [0.5, 0.6],
            "mask_coverage": [0.25, 0.25],
        }
    )

    per_group = aggregate_lung_focus_by_correctness(summary)

    assert list(per_group.index) == ["correct"]


def test_save_correctness_lung_focus_report_writes_csv_and_png(
    manifest_with_masks: pd.DataFrame, tmp_path: Path
) -> None:
    model = build_transfer_model(SMALL)
    summary = summarize_lung_focus(model, manifest_with_masks, image_size=(64, 64))

    csv_path, png_path = save_correctness_lung_focus_report(summary, tmp_path, "test_model")

    assert csv_path.exists()
    assert png_path.exists()
