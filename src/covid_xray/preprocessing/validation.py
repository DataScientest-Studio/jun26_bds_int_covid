from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..config import ARRAY_DIR, SPLITS
from .dataset import ArrayDataset, array_paths


@dataclass(frozen=True)
class SplitComparison:
    split: str
    fresh_samples: int
    class_counts: Dict[str, int]
    has_previous: bool
    previous_samples: Optional[int] = None
    same_shape: Optional[bool] = None
    same_labels: Optional[bool] = None
    same_values: Optional[bool] = None
    max_absolute_difference: Optional[float] = None
    changed_samples: Optional[int] = None

    @property
    def matches(self) -> bool:
        return bool(
            self.has_previous
            and self.same_shape
            and self.same_labels
            and self.same_values
        )


@dataclass(frozen=True)
class CountCheck:
    processed_images: int
    manifest_rows: int
    fresh_samples: int

    @property
    def manifest_matches_processed(self) -> bool:
        return self.processed_images == self.manifest_rows

    @property
    def fresh_matches_manifest(self) -> bool:
        return self.fresh_samples == self.manifest_rows


def check_counts(
    processed_images: int, manifest_rows: int, fresh_samples: int
) -> CountCheck:
    return CountCheck(
        processed_images=processed_images,
        manifest_rows=manifest_rows,
        fresh_samples=fresh_samples,
    )


def compare_with_saved(
    split: str, dataset: ArrayDataset, directory: Path | str = ARRAY_DIR
) -> SplitComparison:
    class_counts = dataset.class_counts()
    x_path, y_path = array_paths(split, directory)

    if not (x_path.exists() and y_path.exists()):
        return SplitComparison(
            split=split,
            fresh_samples=dataset.n_samples,
            class_counts=class_counts,
            has_previous=False,
        )

    X_old, y_old = np.load(x_path), np.load(y_path)
    same_shape = X_old.shape == dataset.X.shape
    same_labels = bool(np.array_equal(y_old, dataset.y))
    same_values = bool(same_shape and np.allclose(X_old, dataset.X))

    max_absolute_difference = None
    changed_samples = None
    if same_shape and not same_values:
        difference = np.abs(X_old - dataset.X)
        max_absolute_difference = float(difference.max())
        changed_samples = int(
            np.any(difference.reshape(len(X_old), -1) > 0, axis=1).sum()
        )

    return SplitComparison(
        split=split,
        fresh_samples=dataset.n_samples,
        class_counts=class_counts,
        has_previous=True,
        previous_samples=int(X_old.shape[0]),
        same_shape=same_shape,
        same_labels=same_labels,
        same_values=same_values,
        max_absolute_difference=max_absolute_difference,
        changed_samples=changed_samples,
    )


def compare_all(
    datasets: Mapping[str, ArrayDataset],
    directory: Path | str = ARRAY_DIR,
    splits: tuple = SPLITS,
) -> List[SplitComparison]:
    return [
        compare_with_saved(split, datasets[split], directory)
        for split in splits
        if split in datasets
    ]


def format_count_check(check: CountCheck) -> str:
    lines = [
        f"Processed folder images: {check.processed_images}",
        f"Manifest images: {check.manifest_rows}",
        f"Fresh total samples: {check.fresh_samples}",
    ]
    if not check.manifest_matches_processed:
        lines.append("WARNING: manifest count does not match processed folder count.")
    if not check.fresh_matches_manifest:
        lines.append("WARNING: fresh sample count does not match manifest count.")
    return "\n".join(lines)


def format_comparison(comparison: SplitComparison) -> str:
    lines = [
        comparison.split.upper(),
        f"  Fresh: {comparison.fresh_samples} samples",
        f"  Classes: {comparison.class_counts}",
    ]

    if not comparison.has_previous:
        lines.append("  No previous save on disk.")
        return "\n".join(lines)

    lines.extend(
        [
            f"  Previous save: {comparison.previous_samples} samples",
            f"  Same shape: {comparison.same_shape}",
            f"  Same labels: {comparison.same_labels}",
            f"  Same pixel values: {comparison.same_values}",
        ]
    )

    if comparison.same_shape and not comparison.same_values:
        lines.append(f"  Max pixel diff: {comparison.max_absolute_difference:.6e}")
        lines.append(
            f"  Changed samples: {comparison.changed_samples} / {comparison.previous_samples}"
        )
    elif not comparison.same_shape:
        delta = comparison.fresh_samples - (comparison.previous_samples or 0)
        lines.append(
            f"  Sample count change: {comparison.previous_samples} -> "
            f"{comparison.fresh_samples} ({delta:+d})"
        )

    return "\n".join(lines)


def format_comparison_report(comparisons: List[SplitComparison]) -> str:
    lines = [format_comparison(comparison) for comparison in comparisons]
    if comparisons and all(comparison.matches for comparison in comparisons):
        lines.append("No data lost - fresh build matches previous save exactly.")
    else:
        lines.append(
            "Differences detected - review counts and diffs above before saving."
        )
    return "\n\n".join(lines)


def comparison_table(comparisons: List[SplitComparison]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": comparison.split,
                "fresh_samples": comparison.fresh_samples,
                "previous_samples": comparison.previous_samples,
                "same_shape": comparison.same_shape,
                "same_labels": comparison.same_labels,
                "same_values": comparison.same_values,
                "changed_samples": comparison.changed_samples,
            }
            for comparison in comparisons
        ]
    )
