from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..config import CLASS_NAMES, ID_TO_LABEL


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    split: str
    report: Dict[str, Any]
    confusion: np.ndarray

    @property
    def accuracy(self) -> float:
        return float(self.report["accuracy"])


def evaluate_model(model, X: np.ndarray, y: np.ndarray, model_name: str, split: str) -> EvaluationResult:
    predictions = model.predict(X)
    label_ids = sorted(ID_TO_LABEL)
    report = classification_report(
        y,
        predictions,
        labels=label_ids,
        target_names=[ID_TO_LABEL[label_id] for label_id in label_ids],
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y, predictions, labels=label_ids)
    return EvaluationResult(model_name=model_name, split=split, report=report, confusion=confusion)


def format_evaluation(result: EvaluationResult) -> str:
    lines = [f"{result.model_name} [{result.split}]", f"  Accuracy: {result.accuracy:.4f}"]
    for class_name in CLASS_NAMES:
        metrics = result.report[class_name]
        lines.append(
            f"  {class_name}: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f} "
            f"support={int(metrics['support'])}"
        )
    macro = result.report["macro avg"]
    lines.append(
        f"  macro avg: precision={macro['precision']:.3f} recall={macro['recall']:.3f} "
        f"f1={macro['f1-score']:.3f}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(result: EvaluationResult, output_path: Path | str) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(6, 5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    image = axis.imshow(result.confusion, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{result.model_name} [{result.split}]")

    for i in range(result.confusion.shape[0]):
        for j in range(result.confusion.shape[1]):
            axis.text(j, i, int(result.confusion[i, j]), ha="center", va="center")

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def save_metrics(results: Mapping[str, EvaluationResult], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {
            "split": result.split,
            "accuracy": result.accuracy,
            "report": result.report,
            "confusion_matrix": result.confusion.tolist(),
        }
        for name, result in results.items()
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path
