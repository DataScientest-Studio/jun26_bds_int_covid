from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import CLASS_NAMES
from .pipeline import BASELINE_REPORTS_DIR


def load_metrics(reports_dir: Path | str = BASELINE_REPORTS_DIR) -> pd.DataFrame:
    """Collect every *_metrics.json in `reports_dir` into one tidy table.

    Filenames follow `{model}[_{region}]_metrics.json`, so the region is
    recovered from the suffix; the unsuffixed files are the full-image runs.
    """
    reports_dir = Path(reports_dir)
    known_regions = ("lungs", "background")

    rows = []
    for path in sorted(reports_dir.glob("*_metrics.json")):
        label = path.name.removesuffix("_metrics.json")
        region = next((r for r in known_regions if label.endswith(f"_{r}")), "full")
        model = label.removesuffix(f"_{region}") if region != "full" else label

        payload = json.loads(path.read_text())
        for split, result in payload.items():
            report = result["report"]
            row = {
                "model": model,
                "region": region,
                "split": split,
                "accuracy": result["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
            }
            row.update(
                {f"f1_{name}": report[name]["f1-score"] for name in CLASS_NAMES if name in report}
            )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "region", "split"]).reset_index(drop=True)


def region_comparison(
    reports_dir: Path | str = BASELINE_REPORTS_DIR,
    split: str = "test",
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pivot: one row per model, one column per region, showing macro F1.

    The `background` column is the confound estimate — how well the class can be
    predicted from pixels that contain no lung tissue at all.
    """
    frame = load_metrics(reports_dir)
    frame = frame[frame["split"] == split]
    if models is not None:
        frame = frame[frame["model"].isin(list(models))]
    return frame.pivot(index="model", columns="region", values="macro_f1")