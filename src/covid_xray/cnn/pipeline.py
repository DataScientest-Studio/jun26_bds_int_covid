from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import tensorflow as tf

from ..config import CLASS_FOLDERS, MODELS_DIR, RAW_DIR, REPORTS_DIR
from ..preprocessing.config import SplitConfig
from ..preprocessing.manifest import Splits, split_manifest, split_summary
from ..training.data import build_raw_manifest
from ..training.evaluation import (
    EvaluationResult,
    format_evaluation,
    plot_confusion_matrix,
    save_metrics,
)
from ..training.pipeline import artifact_name
from .config import CNNConfig
from .dataset import build_datasets, compute_class_weights
from .evaluation import evaluate_dataset
from .model import build_model

CNN_REPORTS_DIR = REPORTS_DIR / "cnn"
EVAL_SPLITS = ("train", "val", "test")
# One artifact name per architecture, so runs never overwrite each other and
# the region suffix stays the only varying part of the filename.
#DEFAULT_MODEL_NAMES = {"scratch": "cnn_scratch", "lenet": "lenet"}
DEFAULT_MODEL_NAMES = {"scratch": "cnn_scratch_v2", "lenet": "lenet"}

def default_model_name(architecture: str) -> str:
    return DEFAULT_MODEL_NAMES[architecture]


@dataclass(frozen=True)
class CNNResult:
    splits: Splits
    region: str = "full"
    architecture: str = "scratch"
    history: Dict[str, List[float]] = field(default_factory=dict)
    evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)
    class_weights: Dict[int, float] = field(default_factory=dict)
    model_path: Optional[Path] = None
    n_parameters: int = 0


def run_cnn(
    raw_dir: Path | str = RAW_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    redundant_csv: Optional[Path | str] = None,
    split_config: SplitConfig = SplitConfig(),
    config: CNNConfig = CNNConfig(),
    reports_dir: Path | str = CNN_REPORTS_DIR,
    models_dir: Path | str = MODELS_DIR,
    model_name: Optional[str] = None,
    save: bool = True,
    verbose: int = 2,
) -> CNNResult:
    """Train the from-scratch CNN under one region condition.

    Reads from RAW_DIR through build_raw_manifest -- the same entry point the
    sklearn baselines use -- so the duplicate-removal step and the split
    membership are identical for a given seed. That is what makes the CNN
    numbers comparable to the hist_gradient_boosting region results rather
    than merely adjacent to them.
    """
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)
    if model_name is None:
        model_name = default_model_name(config.architecture)

    manifest = build_raw_manifest(
        raw_dir=raw_dir, class_folders=class_folders, redundant_csv=redundant_csv
    )
    # Split before building datasets so every region run sees exactly the same
    # train/val/test membership.
    splits = split_manifest(manifest, split_config)
    datasets = build_datasets(splits, config)

    tf.keras.utils.set_random_seed(config.random_state)
    model = build_model(config)

    class_weights = compute_class_weights(splits.train) if config.class_weight else {}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / f"{artifact_name(model_name, config.region)}_ckpt.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=config.epochs,
        class_weight=class_weights or None,
        callbacks=callbacks,
        # The dataset already shuffles; letting Keras shuffle too would only
        # reorder whole batches and costs a buffer for nothing.
        shuffle=False,
        verbose=verbose,
    )

    label = artifact_name(model_name, config.region)
    evaluations = {
        split_name: evaluate_dataset(model, datasets[split_name], label, split_name)
        for split_name in EVAL_SPLITS
    }

    model_path = None
    if save:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{label}.keras"
        model.save(model_path)

        for split_name, result in evaluations.items():
            plot_confusion_matrix(
                result, reports_dir / f"{label}_{split_name}_confusion_matrix.png"
            )
        save_metrics(evaluations, reports_dir / f"{label}_metrics.json")

    return CNNResult(
        splits=splits,
        region=config.region,
        architecture=config.architecture,
        history={key: [float(v) for v in values] for key, values in history.history.items()},
        evaluations=evaluations,
        class_weights=class_weights,
        model_path=model_path,
        n_parameters=int(model.count_params()),
    )


def format_cnn_report(result: CNNResult) -> str:
    sections = [
        f"architecture: {result.architecture}",
        f"region: {result.region}",
        f"parameters: {result.n_parameters:,}",
        split_summary(result.splits).to_string(),
    ]
    if result.class_weights:
        weights = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(result.class_weights.items()))
        sections.append(f"class weights: {weights}")
    for split_name in EVAL_SPLITS:
        if split_name in result.evaluations:
            sections.append(format_evaluation(result.evaluations[split_name]))
    if result.model_path is not None:
        sections.append(f"Saved model: {result.model_path}")
    return "\n\n".join(sections)
