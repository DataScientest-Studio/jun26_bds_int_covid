from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ..config import ID_TO_LABEL
from ..training.evaluation import EvaluationResult


def predict_dataset(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect labels and predictions by iterating the dataset once.

    Labels are pulled from the same iteration as the predictions rather than
    from the manifest, so a shuffled or dropped batch can never silently
    misalign y_true against y_pred.
    """
    y_true = []
    y_pred = []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probabilities, axis=1))
        y_true.append(labels.numpy())
    if not y_true:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_dataset(
    model: tf.keras.Model, dataset: tf.data.Dataset, model_name: str, split: str
) -> EvaluationResult:
    y_true, y_pred = predict_dataset(model, dataset)
    label_ids = sorted(ID_TO_LABEL)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=[ID_TO_LABEL[label_id] for label_id in label_ids],
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=label_ids)
    return EvaluationResult(
        model_name=model_name, split=split, report=report, confusion=confusion
    )
