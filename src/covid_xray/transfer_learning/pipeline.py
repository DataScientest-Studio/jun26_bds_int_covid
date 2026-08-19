from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import tensorflow as tf

from ..config import CLASS_FOLDERS, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from ..preprocessing.config import SplitConfig
from ..preprocessing.manifest import Splits, build_manifest, split_manifest, split_summary
from ..training.evaluation import (
    EvaluationResult,
    format_evaluation,
    plot_confusion_matrix,
    save_metrics,
)
from .config import TransferConfig
from .dataset import build_datasets
from .evaluation import evaluate_dataset
from .model import build_transfer_model

TRANSFER_REPORTS_DIR = REPORTS_DIR / "transfer_learning"
EVAL_SPLITS = ("train", "val", "test")
DEFAULT_MODEL_NAME = "transfer_efficientnetb0"


@dataclass(frozen=True)
class TransferResult:
    splits: Splits
    history: Dict[str, List[float]]
    evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)
    model_path: Optional[Path] = None


def run_transfer_learning(
    processed_dir: Path | str = PROCESSED_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    split_config: SplitConfig = SplitConfig(),
    config: TransferConfig = TransferConfig(),
    reports_dir: Path | str = TRANSFER_REPORTS_DIR,
    models_dir: Path | str = MODELS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    save: bool = True,
    verbose: int = 2,
) -> TransferResult:
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)

    manifest = build_manifest(processed_dir=processed_dir, class_folders=class_folders)
    splits = split_manifest(manifest, split_config)
    datasets = build_datasets(splits, config)

    tf.keras.utils.set_random_seed(config.random_state)
    model = build_transfer_model(config)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=config.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=verbose,
    )

    evaluations = {
        split_name: evaluate_dataset(model, datasets[split_name], model_name, split_name)
        for split_name in EVAL_SPLITS
    }

    model_path = None
    if save:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.keras"
        model.save(model_path)

        for split_name, result in evaluations.items():
            plot_confusion_matrix(
                result, reports_dir / f"{model_name}_{split_name}_confusion_matrix.png"
            )
        save_metrics(evaluations, reports_dir / f"{model_name}_metrics.json")

    return TransferResult(
        splits=splits,
        history=history.history,
        evaluations=evaluations,
        model_path=model_path,
    )


def format_transfer_report(result: TransferResult) -> str:
    sections = [split_summary(result.splits).to_string()]
    for split_name in EVAL_SPLITS:
        if split_name in result.evaluations:
            sections.append(format_evaluation(result.evaluations[split_name]))
    if result.model_path is not None:
        sections.append(f"Saved model: {result.model_path}")
    return "\n\n".join(sections)
