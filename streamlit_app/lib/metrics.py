import json
from pathlib import Path
from typing import Optional

import pandas as pd


def load_metrics_json(path: Path) -> Optional[dict]:
    if not Path(path).exists():
        return None
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def split_summary(metrics: Optional[dict], split: str = "test") -> Optional[dict]:
    if not metrics or split not in metrics:
        return None
    try:
        report = metrics[split]["report"]
        return {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
    except (KeyError, TypeError):
        return None


def per_class_table(metrics: Optional[dict], split: str, class_names) -> Optional[pd.DataFrame]:
    if not metrics or split not in metrics:
        return None
    try:
        report = metrics[split]["report"]
    except (KeyError, TypeError):
        return None
    if not isinstance(report, dict):
        return None
    rows = []
    for class_name in class_names:
        entry = report.get(class_name)
        if not isinstance(entry, dict):
            continue
        try:
            rows.append(
                {
                    "Class": class_name,
                    "Precision": entry["precision"],
                    "Recall": entry["recall"],
                    "F1-score": entry["f1-score"],
                    "Support": int(entry["support"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return None
    return pd.DataFrame(rows)


def confusion_matrix_frame(metrics: Optional[dict], split: str, class_names) -> Optional[pd.DataFrame]:
    if not metrics or split not in metrics:
        return None
    try:
        matrix = metrics[split]["confusion_matrix"]
        return pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))
    except (KeyError, TypeError, ValueError):
        return None


def build_comparison_table(entries, split: str = "test") -> pd.DataFrame:
    rows = []
    for entry in entries:
        metrics_path = entry.get("metrics_path")
        if not metrics_path:
            continue
        metrics = load_metrics_json(metrics_path)
        summary = split_summary(metrics, split=split)
        if summary is None:
            continue
        rows.append(
            {
                "Model": entry.get("name", "Unnamed model"),
                "Family": entry.get("family", ""),
                "Test accuracy": summary["accuracy"],
                "Macro F1": summary["macro_f1"],
            }
        )
    return pd.DataFrame(rows)
