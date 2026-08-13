from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

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



def build_hist_gradient_boosting(
    config: BaselineConfig = BaselineConfig(),
) -> HistGradientBoostingClassifier:
    # Trees express pixel *combinations* ("dark here AND bright there"), which a
    # linear model cannot. Same flattened input as logistic regression.
    return HistGradientBoostingClassifier(
        max_iter=config.hgb_max_iter,
        learning_rate=config.hgb_learning_rate,
        max_leaf_nodes=config.hgb_max_leaf_nodes,
        class_weight=config.class_weight,  # requires scikit-learn >= 1.4
        early_stopping=True,
        random_state=config.random_state,
    )


def build_baseline_models(config: BaselineConfig = BaselineConfig()) -> dict:
    return {
        "dummy": build_dummy_classifier(config),
        "logistic_regression": build_logistic_regression(config),
        "hist_gradient_boosting": build_hist_gradient_boosting(config),
    }

