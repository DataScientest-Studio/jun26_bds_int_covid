import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.paths import REPORTS_DIR


@st.cache_data
def load_metrics(relative_path: str) -> dict:
    with (REPORTS_DIR / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_summary(relative_path: str) -> dict:
    test = load_metrics(relative_path)["test"]
    report = test["report"]
    return {
        "accuracy": float(test["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "covid_recall": float(report["COVID"]["recall"]),
    }


def comparison_frame() -> pd.DataFrame:
    models = {
        "Logistic regression": "baseline/logistic_regression_metrics.json",
        "Simple CNN": "cnn/cnn_simple_metrics.json",
        "EfficientNetB0": "transfer_learning/transfer_efficientnetb0_metrics.json",
        "Fine-tuned EfficientNetB0": "transfer_learning/transfer_efficientnetb0_finetuned_metrics.json",
    }
    rows = []
    for name, path in models.items():
        metrics = test_summary(path)
        rows.append({"Model": name, "Accuracy": metrics["accuracy"], "Macro F1": metrics["macro_f1"]})
    return pd.DataFrame(rows)


def class_metrics(relative_path: str) -> pd.DataFrame:
    report = load_metrics(relative_path)["test"]["report"]
    classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    rows = []
    for name in classes:
        values = report[name]
        rows.append(
            {
                "Class": name.replace("_", " "),
                "Precision": float(values["precision"]),
                "Recall": float(values["recall"]),
                "F1": float(values["f1-score"]),
            }
        )
    return pd.DataFrame(rows)
