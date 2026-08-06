from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import sklearn

from ..config import (
    ARRAY_DIR,
    CLASS_FOLDERS,
    LABEL_TO_ID,
    PROCESSED_DIR,
    RAW_DIR,
)
from .augmentation import AugmentConfig
from .config import PreprocessConfig, SplitConfig
from .dataset import (
    ArrayDataset,
    build_split_arrays,
    format_failures,
    save_arrays,
)
from .files import (
    CopyResult,
    copy_dataset,
    count_images_and_masks,
    format_copy_report,
    load_redundant_files,
)
from .manifest import Splits, build_manifest, split_manifest, split_summary
from .validation import (
    CountCheck,
    SplitComparison,
    check_counts,
    compare_all,
    format_comparison_report,
    format_count_check,
)

METADATA_FILENAME = "preprocessing.json"


@dataclass(frozen=True)
class PreprocessingResult:
    manifest: pd.DataFrame
    splits: Splits
    datasets: Dict[str, ArrayDataset]
    counts: CountCheck
    comparisons: List[SplitComparison]
    metadata: Dict[str, Any]
    copy_results: Dict[str, CopyResult] = field(default_factory=dict)
    saved_paths: Dict[str, Tuple[Path, Path]] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return sum(dataset.n_samples for dataset in self.datasets.values())

    @property
    def failed(self) -> List[Tuple[str, str]]:
        return [
            failure
            for dataset in self.datasets.values()
            for failure in dataset.failed
        ]


def package_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "opencv-python": cv2.__version__,
    }


def build_metadata(
    datasets: Mapping[str, ArrayDataset],
    raw_dir: Path,
    processed_dir: Path,
    array_dir: Path,
    split_config: SplitConfig,
    preprocess_config: PreprocessConfig,
    augment_config: AugmentConfig,
    augment_splits: Tuple[str, ...],
) -> Dict[str, Any]:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "array_dir": str(array_dir),
        "label_to_id": dict(LABEL_TO_ID),
        "split_config": asdict(split_config),
        "preprocess_config": asdict(preprocess_config),
        "augment_config": asdict(augment_config),
        "augment_splits": list(augment_splits),
        "samples": {
            split: dataset.n_samples for split, dataset in datasets.items()
        },
        "class_counts": {
            split: dataset.class_counts() for split, dataset in datasets.items()
        },
        "failed": {
            split: len(dataset.failed) for split, dataset in datasets.items()
        },
        "versions": package_versions(),
    }
    return json.loads(json.dumps(metadata, sort_keys=True))


def save_metadata(
    metadata: Mapping[str, Any], directory: Path | str = ARRAY_DIR
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def load_metadata(directory: Path | str = ARRAY_DIR) -> Dict[str, Any]:
    path = Path(directory) / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No preprocessing metadata found at {path}")
    return json.loads(path.read_text())


def run_preprocessing(
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    array_dir: Path | str = ARRAY_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    redundant_csv: Optional[Path | str] = None,
    split_config: SplitConfig = SplitConfig(),
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    augment_config: AugmentConfig = AugmentConfig(),
    augment_splits: Tuple[str, ...] = ("train",),
    copy_raw_files: bool = True,
    save: bool = True,
) -> PreprocessingResult:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    array_dir = Path(array_dir)

    copy_results: Dict[str, CopyResult] = {}
    if copy_raw_files:
        redundant_files = (
            load_redundant_files(redundant_csv) if redundant_csv is not None else set()
        )
        copy_results = copy_dataset(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            class_folders=class_folders,
            redundant_files=redundant_files,
        )

    manifest = build_manifest(processed_dir, class_folders)
    splits = split_manifest(manifest, split_config)
    datasets = build_split_arrays(
        splits,
        config=preprocess_config,
        augment_config=augment_config,
        augment_splits=augment_splits,
    )

    counts = check_counts(
        processed_images=count_images_and_masks(processed_dir, class_folders)["images"],
        manifest_rows=len(manifest),
        fresh_samples=sum(dataset.n_samples for dataset in datasets.values()),
    )
    comparisons = compare_all(datasets, array_dir)
    metadata = build_metadata(
        datasets=datasets,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        array_dir=array_dir,
        split_config=split_config,
        preprocess_config=preprocess_config,
        augment_config=augment_config,
        augment_splits=augment_splits,
    )

    saved_paths: Dict[str, Tuple[Path, Path]] = {}
    if save:
        saved_paths = save_arrays(datasets, array_dir)
        save_metadata(metadata, array_dir)

    return PreprocessingResult(
        manifest=manifest,
        splits=splits,
        datasets=datasets,
        counts=counts,
        comparisons=comparisons,
        metadata=metadata,
        copy_results=copy_results,
        saved_paths=saved_paths,
    )


def format_report(result: PreprocessingResult) -> str:
    sections = []
    if result.copy_results:
        sections.append(format_copy_report(result.copy_results))

    sections.append(split_summary(result.splits).to_string())
    sections.append(format_count_check(result.counts))

    failures = [
        format_failures(dataset)
        for dataset in result.datasets.values()
        if dataset.failed
    ]
    sections.extend(failures)

    sections.append(format_comparison_report(result.comparisons))

    if result.saved_paths:
        saved = ", ".join(sorted(result.saved_paths))
        sections.append(f"Saved arrays for: {saved}")

    return "\n\n".join(sections)
