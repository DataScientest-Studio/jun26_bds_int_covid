from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES
from .history import HistoryDict, load_history, plot_history_comparison
from .pipeline import TRANSFER_REPORTS_DIR


def load_metrics(reports_dir: Path | str = TRANSFER_REPORTS_DIR) -> pd.DataFrame:
    """Collect every `*_metrics.json` in `reports_dir` into one tidy table.

    One row per (model, split), with accuracy, macro F1, and per-class F1, so
    different transfer-learning runs (e.g. augmented vs. class-weighted vs.
    masked) can be compared side by side.
    """
    reports_dir = Path(reports_dir)

    rows = []
    for path in sorted(reports_dir.glob("*_metrics.json")):
        model = path.name.removesuffix("_metrics.json")
        payload = json.loads(path.read_text())
        for split, result in payload.items():
            report = result["report"]
            row = {
                "model": model,
                "split": split,
                "accuracy": result["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
            }
            row.update(
                {f"f1_{name}": report[name]["f1-score"] for name in CLASS_NAMES if name in report}
            )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)


def model_comparison(
    reports_dir: Path | str = TRANSFER_REPORTS_DIR,
    split: str = "test",
    models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """One row per model, showing accuracy/macro F1/per-class F1 on `split`."""
    frame = load_metrics(reports_dir)
    frame = frame[frame["split"] == split].drop(columns=["split"])
    if models is not None:
        frame = frame[frame["model"].isin(list(models))]
    return frame.set_index("model").sort_index()


def load_histories(
    model_names: Iterable[str], reports_dir: Path | str = TRANSFER_REPORTS_DIR
) -> Dict[str, HistoryDict]:
    """Load `{model_name}_history.json` for every requested model that has one."""
    reports_dir = Path(reports_dir)
    histories = {}
    for name in model_names:
        history_path = reports_dir / f"{name}_history.json"
        if history_path.exists():
            histories[name] = load_history(history_path)
    return histories


def plot_test_comparison(
    labels: Mapping[str, str],
    output_path: Path | str,
    reports_dir: Path | str = TRANSFER_REPORTS_DIR,
    split: str = "test",
    title: str = "EfficientNetB0 transfer-learning runs compared (test split)",
    y_min: float = 0.60,
) -> Path:
    """Bar chart of accuracy vs. macro F1 across several runs, in a fixed order.

    `labels` maps `model_name -> short display label` (e.g. `"masked": "masked\\n(lungs only)"`),
    so the chart can control run order independently of alphabetical model names.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    frame = model_comparison(reports_dir=reports_dir, split=split, models=labels)
    frame = frame.reindex(list(labels))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(frame))
    figure = Figure(figsize=(10.5, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    for offset, column, color, legend_label in (
        (-0.2, "accuracy", "#1f77b4", "Accuracy"),
        (0.2, "macro_f1", "#ff7f0e", "Macro F1"),
    ):
        bars = axis.bar(positions + offset, frame[column], width=0.4, color=color, label=legend_label)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)

    axis.set_xticks(positions)
    axis.set_xticklabels([labels[name] for name in frame.index])
    axis.set_ylabel(f"Score ({split} split)")
    axis.set_ylim(y_min, 1.0)
    axis.set_title(title)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def plot_metric_across_runs(
    model_names: Iterable[str],
    metric: str,
    output_path: Path | str,
    reports_dir: Path | str = TRANSFER_REPORTS_DIR,
    title: Optional[str] = None,
) -> Path:
    """Overlay one training-history metric (e.g. `val_loss`) across several runs."""
    histories = load_histories(model_names, reports_dir)
    return plot_history_comparison(histories, metric, output_path, title=title)
