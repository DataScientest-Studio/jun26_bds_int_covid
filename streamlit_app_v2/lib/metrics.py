import json
from pathlib import Path

import pandas as pd


def load_metrics(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def summary(path: Path, split: str = "test") -> dict:
    report = load_metrics(path).get(split, {}).get("report", {})
    return {
        "accuracy": report.get("accuracy"),
        "macro_f1": report.get("macro avg", {}).get("f1-score"),
        "macro_precision": report.get("macro avg", {}).get("precision"),
        "macro_recall": report.get("macro avg", {}).get("recall"),
    }


def class_score(path: Path, class_name: str, metric: str = "recall", split: str = "test"):
    return load_metrics(path).get(split, {}).get("report", {}).get(class_name, {}).get(metric)


def comparison(entries: list[tuple[str, Path]], split: str = "test") -> pd.DataFrame:
    rows = []
    for label, path in entries:
        values = summary(path, split)
        if values["accuracy"] is None:
            continue
        rows.append(
            {
                "Model": label,
                "Accuracy": values["accuracy"],
                "Macro F1": values["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def percent(value, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def points(value, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:+.{digits}f} pts"

