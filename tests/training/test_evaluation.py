from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from covid_xray.config import CLASS_NAMES
from covid_xray.training import (
    BaselineConfig,
    build_logistic_regression,
    evaluate_model,
    format_evaluation,
    load_flat_features,
    plot_confusion_matrix,
    save_metrics,
)

CONFIG = BaselineConfig(image_size=(8, 8))


def fitted_model_and_features(manifest: pd.DataFrame):
    X, y = load_flat_features(manifest, CONFIG)
    model = build_logistic_regression(CONFIG)
    model.fit(X, y)
    return model, X, y


def test_evaluate_model_reports_every_known_class(manifest: pd.DataFrame) -> None:
    model, X, y = fitted_model_and_features(manifest)

    result = evaluate_model(model, X, y, "logistic_regression", "train")

    assert result.model_name == "logistic_regression"
    assert result.split == "train"
    assert set(CLASS_NAMES) <= set(result.report)
    assert result.confusion.shape == (len(CLASS_NAMES), len(CLASS_NAMES))
    assert 0.0 <= result.accuracy <= 1.0


def test_format_evaluation_includes_accuracy_and_classes(manifest: pd.DataFrame) -> None:
    model, X, y = fitted_model_and_features(manifest)
    result = evaluate_model(model, X, y, "logistic_regression", "train")

    text = format_evaluation(result)

    assert "Accuracy" in text
    for class_name in CLASS_NAMES:
        assert class_name in text


def test_plot_confusion_matrix_writes_a_file(manifest: pd.DataFrame, tmp_path: Path) -> None:
    model, X, y = fitted_model_and_features(manifest)
    result = evaluate_model(model, X, y, "logistic_regression", "train")

    output_path = plot_confusion_matrix(result, tmp_path / "confusion.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_metrics_writes_expected_structure(manifest: pd.DataFrame, tmp_path: Path) -> None:
    model, X, y = fitted_model_and_features(manifest)
    result = evaluate_model(model, X, y, "logistic_regression", "train")

    output_path = save_metrics({"logistic_regression": result}, tmp_path / "metrics.json")

    payload = json.loads(output_path.read_text())
    assert "logistic_regression" in payload
    assert payload["logistic_regression"]["accuracy"] == result.accuracy
    assert payload["logistic_regression"]["split"] == "train"
