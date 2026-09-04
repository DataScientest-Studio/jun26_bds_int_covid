from __future__ import annotations

from typing import List

import tensorflow as tf
from sklearn.metrics import f1_score

from .evaluation import predict_dataset


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Tracks macro-averaged F1 on `dataset` after every epoch.

    Keras has no built-in multi-class F1 metric, so this runs a prediction
    pass over `dataset` at `on_epoch_end` and keeps the scores in `self.scores`
    (one entry per completed epoch) for the caller to merge into the training
    history after `fit` finishes.
    """

    def __init__(self, dataset: tf.data.Dataset, name: str = "macro_f1") -> None:
        super().__init__()
        self.dataset = dataset
        self.name = name
        self.scores: List[float] = []

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        y_true, y_pred = predict_dataset(self.model, self.dataset)
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        self.scores.append(float(score))
