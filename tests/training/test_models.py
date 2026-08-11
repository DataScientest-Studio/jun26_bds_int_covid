from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from covid_xray.training import (
    BaselineConfig,
    build_baseline_models,
    build_dummy_classifier,
    build_logistic_regression,
)

X = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [0.9, 0.9]], dtype=np.float32)
Y = np.array([0, 0, 1, 1])


def test_build_dummy_classifier_predicts_majority_class() -> None:
    model = build_dummy_classifier(BaselineConfig())
    model.fit(X, np.array([0, 0, 0, 1]))

    assert isinstance(model, DummyClassifier)
    assert set(model.predict(X)) == {0}


def test_build_logistic_regression_fits_and_predicts() -> None:
    model = build_logistic_regression(BaselineConfig())
    model.fit(X, Y)

    assert isinstance(model, LogisticRegression)
    assert model.predict(X).shape == Y.shape


def test_build_baseline_models_returns_dummy_and_logistic_regression() -> None:
    models = build_baseline_models(BaselineConfig())

    assert set(models) == {"dummy", "logistic_regression"}
    assert isinstance(models["dummy"], DummyClassifier)
    assert isinstance(models["logistic_regression"], LogisticRegression)
