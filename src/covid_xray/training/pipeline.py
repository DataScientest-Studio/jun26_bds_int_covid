from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import joblib

from ..config import CLASS_FOLDERS, MODELS_DIR, RAW_DIR, REPORTS_DIR
from ..preprocessing.config import SplitConfig
from ..preprocessing.manifest import Splits, split_manifest, split_summary
from .config import BaselineConfig
from .data import build_raw_manifest
from .evaluation import (
    EvaluationResult,
    evaluate_model,
    format_evaluation,
    plot_confusion_matrix,
    save_metrics,
)
from .features import load_flat_features
from .models import build_baseline_models

BASELINE_REPORTS_DIR = REPORTS_DIR / "baseline"
EVAL_SPLITS = ("train", "val", "test")


def artifact_name(model_name: str, region: str) -> str:
    """Suffix artifacts with the region so runs never overwrite each other.

    The full-image run keeps its original filenames, so previously saved
    baselines stay valid.
    """
    return model_name if region == "full" else f"{model_name}_{region}"


@dataclass(frozen=True)
class BaselineResult:
    splits: Splits
    region: str = "full"
    evaluations: Dict[str, Dict[str, EvaluationResult]] = field(default_factory=dict)
    model_paths: Dict[str, Path] = field(default_factory=dict)


def run_baseline(
    raw_dir: Path | str = RAW_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    redundant_csv: Optional[Path | str] = None,
    split_config: SplitConfig = SplitConfig(),
    config: BaselineConfig = BaselineConfig(),
    reports_dir: Path | str = BASELINE_REPORTS_DIR,
    models_dir: Path | str = MODELS_DIR,
    model_names: Optional[Iterable[str]] = None,
    save: bool = True,
) -> BaselineResult:
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)

    manifest = build_raw_manifest(
        raw_dir=raw_dir, class_folders=class_folders, redundant_csv=redundant_csv
    )
    # Split first, then extract features, so every region run sees exactly the
    # same train/test membership for a given seed.
    splits = split_manifest(manifest, split_config)

    features = {
        split_name: load_flat_features(splits[split_name], config)
        for split_name in EVAL_SPLITS
    }

    models = build_baseline_models(config)
    if model_names is not None:
        requested = list(model_names)
        unknown = [name for name in requested if name not in models]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(models)}")
        models = {name: models[name] for name in requested}

    evaluations: Dict[str, Dict[str, EvaluationResult]] = {}
    model_paths: Dict[str, Path] = {}

    for name, model in models.items():
        X_train, y_train = features["train"]
        model.fit(X_train, y_train)

        label = artifact_name(name, config.region)
        evaluations[name] = {
            split_name: evaluate_model(model, X, y, label, split_name)
            for split_name, (X, y) in features.items()
        }

        if save:
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"baseline_{label}.joblib"
            joblib.dump(model, model_path)
            model_paths[name] = model_path

    if save:
        for name, split_evals in evaluations.items():
            label = artifact_name(name, config.region)
            save_metrics(split_evals, reports_dir / f"{label}_metrics.json")
            for split_name, result in split_evals.items():
                plot_confusion_matrix(
                    result, reports_dir / f"{label}_{split_name}_confusion_matrix.png"
                )

    return BaselineResult(
        splits=splits,
        region=config.region,
        evaluations=evaluations,
        model_paths=model_paths,
    )


def format_baseline_report(result: BaselineResult) -> str:
    sections = [f"region: {result.region}", split_summary(result.splits).to_string()]
    for split_evals in result.evaluations.values():
        for split_name in EVAL_SPLITS:
            if split_name in split_evals:
                sections.append(format_evaluation(split_evals[split_name]))
    if result.model_paths:
        saved = ", ".join(f"{name} -> {path}" for name, path in result.model_paths.items())
        sections.append(f"Saved models: {saved}")
    return "\n\n".join(sections)
