from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from .config import BaselineConfig

DUMMY_STRATEGY = "most_frequent"


def build_dummy_classifier(config: BaselineConfig = BaselineConfig()) -> DummyClassifier:
    return DummyClassifier(strategy=DUMMY_STRATEGY, random_state=config.random_state)


def build_logistic_regression(config: BaselineConfig = BaselineConfig()) -> LogisticRegression:
    return LogisticRegression(
        max_iter=config.logreg_max_iter,
        C=config.logreg_C,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )


def build_baseline_models(config: BaselineConfig = BaselineConfig()) -> dict:
    return {
        "dummy": build_dummy_classifier(config),
        "logistic_regression": build_logistic_regression(config),
    }
