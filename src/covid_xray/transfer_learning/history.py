from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

HistoryDict = Dict[str, List[float]]

PANELS: Tuple[Tuple[str, str, str, str], ...] = (
    ("loss", "val_loss", "Loss", "loss"),
    ("accuracy", "val_accuracy", "Accuracy", "accuracy"),
    ("train_macro_f1", "val_macro_f1", "Macro F1", "macro_f1"),
)


def save_history(history: Mapping[str, Sequence[float]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: list(values) for key, values in history.items()}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def load_history(path: Path | str) -> HistoryDict:
    return json.loads(Path(path).read_text())


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    output_path: Path | str,
    model_name: str = "",
) -> Path:
    """Plot train-vs-validation curves for loss, accuracy, and macro F1.

    Panels are skipped when neither the train nor the validation key for that
    metric is present in `history` (e.g. macro F1 tracking is optional).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (train_key, val_key, title)
        for train_key, val_key, title, _ in PANELS
        if train_key in history or val_key in history
    ]
    if not panels:
        panels = [("loss", "val_loss", "Loss")]

    epochs = list(range(1, len(next(iter(history.values()), [])) + 1))

    figure = Figure(figsize=(5.5 * len(panels), 4.2))
    FigureCanvasAgg(figure)

    for index, (train_key, val_key, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), index)
        if train_key in history:
            train_epochs = range(1, len(history[train_key]) + 1)
            axis.plot(train_epochs, history[train_key], label="train", marker="o", markersize=3)
        if val_key in history:
            val_epochs = range(1, len(history[val_key]) + 1)
            axis.plot(val_epochs, history[val_key], label="val", marker="o", markersize=3)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.3)

    figure.suptitle(model_name or "Training history")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def plot_history_comparison(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    metric: str,
    output_path: Path | str,
    title: str | None = None,
) -> Path:
    """Overlay one metric (e.g. `val_loss`, `val_macro_f1`) across several runs.

    `histories` maps a run label (e.g. model name) to its history dict, so
    different trainings can be compared directly on the same axes.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(7, 5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    for label, history in histories.items():
        if metric not in history:
            continue
        values = history[metric]
        axis.plot(range(1, len(values) + 1), values, label=label, marker="o", markersize=3)

    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.set_title(title or f"{metric} across runs")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path
