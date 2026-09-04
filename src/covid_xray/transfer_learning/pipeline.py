from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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
from .callbacks import MacroF1Callback
from .config import TransferConfig
from .dataset import build_datasets, compute_balanced_class_weights
from .evaluation import evaluate_dataset
from .history import load_history, plot_training_history, save_history
from .model import build_transfer_model, prepare_for_fine_tuning

TRANSFER_REPORTS_DIR = REPORTS_DIR / "transfer_learning"
EVAL_SPLITS = ("train", "val", "test")
DEFAULT_MODEL_NAME = "transfer_efficientnetb0"
CHECKPOINTS_DIRNAME = "checkpoints"


def checkpoint_dir_for(models_dir: Path | str, model_name: str) -> Path:
    return Path(models_dir) / CHECKPOINTS_DIRNAME / model_name


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob("epoch_*.keras"))
    return checkpoints[-1] if checkpoints else None


def _epoch_from_checkpoint(path: Path) -> int:
    """Recover the next 0-indexed epoch to train from a checkpoint filename.

    Keras's `ModelCheckpoint` formats the `{epoch}` filepath placeholder as
    the 1-indexed epoch number just completed (e.g. `epoch_0001.keras` after
    finishing the first epoch), which conveniently equals the 0-indexed
    epoch to resume from.
    """
    return int(path.stem.removeprefix("epoch_"))


def _merge_histories(
    left: Dict[str, List[float]], right: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    merged_keys = set(left) | set(right)
    return {
        key: list(left.get(key, [])) + list(right.get(key, [])) for key in merged_keys
    }


def _build_callbacks(
    config: TransferConfig,
    datasets: Mapping[str, tf.data.Dataset],
    checkpoints_dir: Path,
    *,
    save: bool,
    early_stopping_patience: int,
    include_reduce_lr: bool,
) -> Tuple[List[tf.keras.callbacks.Callback], MacroF1Callback, Optional[MacroF1Callback]]:
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    if config.save_checkpoints and save:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoints_dir / "epoch_{epoch:04d}.keras"),
                save_freq="epoch",
            )
        )
    if include_reduce_lr and config.reduce_lr_on_plateau:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.reduce_lr_min_lr,
            )
        )

    val_f1_callback = MacroF1Callback(datasets["val"], name="val_macro_f1")
    callbacks.append(val_f1_callback)
    train_f1_callback = None
    if config.track_train_f1:
        train_f1_callback = MacroF1Callback(datasets["train"], name="train_macro_f1")
        callbacks.append(train_f1_callback)
    return callbacks, val_f1_callback, train_f1_callback


def _fit_and_collect_history(
    model: tf.keras.Model,
    datasets: Mapping[str, tf.data.Dataset],
    config: TransferConfig,
    *,
    epochs: int,
    initial_epoch: int,
    callbacks: List[tf.keras.callbacks.Callback],
    val_f1_callback: MacroF1Callback,
    train_f1_callback: Optional[MacroF1Callback],
    class_weight: Optional[Dict[int, float]],
    verbose: int,
) -> Dict[str, List[float]]:
    if initial_epoch >= epochs:
        return {}

    fit_history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=False,
        verbose=verbose,
    )
    history: Dict[str, List[float]] = dict(fit_history.history)
    history["val_macro_f1"] = val_f1_callback.scores
    if train_f1_callback is not None:
        history["train_macro_f1"] = train_f1_callback.scores
    return history


@dataclass(frozen=True)
class TransferResult:
    splits: Splits
    history: Dict[str, List[float]] = field(default_factory=dict)
    evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)
    model_path: Optional[Path] = None
    history_path: Optional[Path] = None


def run_transfer_learning(
    processed_dir: Path | str = PROCESSED_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    split_config: SplitConfig = SplitConfig(),
    config: TransferConfig = TransferConfig(),
    reports_dir: Path | str = TRANSFER_REPORTS_DIR,
    models_dir: Path | str = MODELS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    save: bool = True,
    resume: bool = False,
    verbose: int = 2,
) -> TransferResult:
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)

    manifest = build_manifest(processed_dir=processed_dir, class_folders=class_folders)
    splits = split_manifest(manifest, split_config)
    datasets = build_datasets(splits, config)

    tf.keras.utils.set_random_seed(config.random_state)

    checkpoints_dir = checkpoint_dir_for(models_dir, model_name)
    initial_epoch = 0
    model = None
    previous_history: Dict[str, List[float]] = {}
    if resume:
        latest_checkpoint = _latest_checkpoint(checkpoints_dir)
        if latest_checkpoint is not None:
            model = tf.keras.models.load_model(latest_checkpoint)
            initial_epoch = _epoch_from_checkpoint(latest_checkpoint)
            history_path = reports_dir / f"{model_name}_history.json"
            if history_path.exists():
                previous_history = load_history(history_path)
    if model is None:
        model = build_transfer_model(config)

    class_weight = (
        compute_balanced_class_weights(splits.train) if config.use_class_weight else None
    )

    phase1_callbacks, val_f1_callback, train_f1_callback = _build_callbacks(
        config,
        datasets,
        checkpoints_dir,
        save=save,
        early_stopping_patience=config.early_stopping_patience,
        include_reduce_lr=False,
    )
    history = _fit_and_collect_history(
        model,
        datasets,
        config,
        epochs=config.epochs,
        initial_epoch=initial_epoch,
        callbacks=phase1_callbacks,
        val_f1_callback=val_f1_callback,
        train_f1_callback=train_f1_callback,
        class_weight=class_weight,
        verbose=verbose,
    )
    if previous_history:
        history = _merge_histories(previous_history, history)

    if config.fine_tune:
        completed_epochs = len(history.get("loss", []))
        model = prepare_for_fine_tuning(model, config)
        phase2_callbacks, val_f1_callback, train_f1_callback = _build_callbacks(
            config,
            datasets,
            checkpoints_dir,
            save=save,
            early_stopping_patience=config.fine_tune_early_stopping_patience,
            include_reduce_lr=True,
        )
        phase2_history = _fit_and_collect_history(
            model,
            datasets,
            config,
            epochs=completed_epochs + config.fine_tune_epochs,
            initial_epoch=completed_epochs,
            callbacks=phase2_callbacks,
            val_f1_callback=val_f1_callback,
            train_f1_callback=train_f1_callback,
            class_weight=class_weight,
            verbose=verbose,
        )
        history = _merge_histories(history, phase2_history)

    evaluations = {
        split_name: evaluate_dataset(model, datasets[split_name], model_name, split_name)
        for split_name in EVAL_SPLITS
    }

    model_path = None
    history_path = None
    if save:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.keras"
        model.save(model_path)

        for split_name, result in evaluations.items():
            plot_confusion_matrix(
                result, reports_dir / f"{model_name}_{split_name}_confusion_matrix.png"
            )
        save_metrics(evaluations, reports_dir / f"{model_name}_metrics.json")

        history_path = save_history(history, reports_dir / f"{model_name}_history.json")
        plot_training_history(
            history, reports_dir / f"{model_name}_history.png", model_name=model_name
        )

    return TransferResult(
        splits=splits,
        history=history,
        evaluations=evaluations,
        model_path=model_path,
        history_path=history_path,
    )


def format_transfer_report(result: TransferResult) -> str:
    sections = [split_summary(result.splits).to_string()]
    for split_name in EVAL_SPLITS:
        if split_name in result.evaluations:
            sections.append(format_evaluation(result.evaluations[split_name]))
    if result.model_path is not None:
        sections.append(f"Saved model: {result.model_path}")
    if result.history_path is not None:
        sections.append(f"Saved history: {result.history_path}")
    return "\n\n".join(sections)
