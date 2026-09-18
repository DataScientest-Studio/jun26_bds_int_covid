from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..config import (
    CLASS_COLUMN,
    CLASS_FOLDERS,
    IMAGE_PATH_COLUMN,
    MASK_PATH_COLUMN,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from ..preprocessing.config import SplitConfig
from ..preprocessing.manifest import Splits, build_manifest, missing_paths, split_manifest
from .config import TransferConfig
from .history import best_checkpoint_path
from .pipeline import TransferResult, checkpoint_dir_for, run_transfer_learning

DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "lung_only_balanced"
DEFAULT_MODEL_NAME = "transfer_efficientnetb0_lung_only_materialized"
MANIFEST_FILENAME = "manifest.csv"
METADATA_FILENAME = "preprocessing.json"


@dataclass(frozen=True)
class LungOnlyPreprocessConfig:
    image_size: Tuple[int, int] = (224, 224)
    rotation_degrees: float = 15.0
    mask_threshold: int = 127
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if min(self.image_size) <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if not 0 <= self.rotation_degrees <= 180:
            raise ValueError("rotation_degrees must be between 0 and 180")
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")


@dataclass(frozen=True)
class LungOnlyDatasetResult:
    output_dir: Path
    manifest_path: Path
    metadata_path: Path
    splits: Splits
    counts: Dict[str, Dict[str, int]]


def _read_grayscale(path: Path, kind: str) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise ValueError(f"Could not read {kind}: {path}")
    return array


def _masked_resized_image(
    image_path: Path,
    mask_path: Path,
    config: LungOnlyPreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    image = _read_grayscale(image_path, "image")
    mask = _read_grayscale(mask_path, "mask")
    if mask.shape != image.shape:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = mask > config.mask_threshold
    masked = np.where(binary, image, 0).astype(np.uint8)
    width, height = config.image_size[1], config.image_size[0]
    masked = cv2.resize(masked, (width, height), interpolation=cv2.INTER_AREA)
    binary_u8 = cv2.resize(
        binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    # Reapply the binary mask after resizing so every saved background pixel is
    # exactly zero, including pixels at the interpolated lung boundary.
    masked[binary_u8 == 0] = 0
    return masked, binary_u8


def _rotate_lung_only(
    image: np.ndarray, mask: np.ndarray, angle: float
) -> np.ndarray:
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated[rotated_mask == 0] = 0
    return rotated


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write processed image: {path}")


def _record(class_name: str, path: Path) -> Dict[str, object]:
    return {
        CLASS_COLUMN: class_name,
        IMAGE_PATH_COLUMN: str(path.resolve()),
        MASK_PATH_COLUMN: "",
    }


def materialize_lung_only_dataset(
    source_dir: Path | str = PROCESSED_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    split_config: SplitConfig = SplitConfig(),
    config: LungOnlyPreprocessConfig = LungOnlyPreprocessConfig(),
) -> LungOnlyDatasetResult:
    """Write leakage-safe, lungs-only model inputs to disk.

    The source data is split before balancing. Train classes are balanced by
    saving rotated copies of randomly sampled minority-class images. Validation
    and test images are masked and resized but are never augmented or balanced.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from source_dir")

    manifest = build_manifest(source_dir, class_folders)
    missing = missing_paths(manifest)
    if missing[IMAGE_PATH_COLUMN] or missing[MASK_PATH_COLUMN]:
        raise FileNotFoundError(
            "Every source image needs a lung mask; missing "
            f"{missing[IMAGE_PATH_COLUMN]} images and {missing[MASK_PATH_COLUMN]} masks"
        )
    source_splits = split_manifest(manifest, split_config)
    rng = np.random.default_rng(config.random_state)
    records: Dict[str, list[Dict[str, object]]] = {
        name: [] for name in ("train", "val", "test")
    }

    def load(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        return _masked_resized_image(
            Path(row[IMAGE_PATH_COLUMN]), Path(row[MASK_PATH_COLUMN]), config
        )

    for split_name, frame in source_splits.items():
        for row_number, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
            image, _ = load(row)
            folder = class_folders[str(row[CLASS_COLUMN])]
            filename = f"{Path(row[IMAGE_PATH_COLUMN]).stem}__orig_{row_number:05d}.png"
            destination = output_dir / split_name / folder / "images" / filename
            _write_png(destination, image)
            records[split_name].append(_record(str(row[CLASS_COLUMN]), destination))

    train_counts = source_splits.train[CLASS_COLUMN].value_counts()
    target_count = int(train_counts.max())
    augmentation_index = 0
    for class_name, folder in class_folders.items():
        group = source_splits.train[source_splits.train[CLASS_COLUMN] == class_name]
        deficit = target_count - len(group)
        if deficit <= 0:
            continue
        sampled_positions = rng.choice(len(group), size=deficit, replace=True)
        for position in sampled_positions:
            row = group.iloc[int(position)]
            image, mask = load(row)
            magnitude = (
                rng.uniform(1.0, config.rotation_degrees)
                if config.rotation_degrees >= 1
                else config.rotation_degrees
            )
            angle = float(magnitude * rng.choice((-1.0, 1.0)))
            rotated = _rotate_lung_only(image, mask, angle)
            filename = (
                f"{Path(row[IMAGE_PATH_COLUMN]).stem}__rot_"
                f"{augmentation_index:06d}_{angle:+06.2f}.png"
            )
            destination = output_dir / "train" / folder / "images" / filename
            _write_png(destination, rotated)
            records["train"].append(_record(class_name, destination))
            augmentation_index += 1

    materialized_splits = Splits(
        train=pd.DataFrame(records["train"]),
        val=pd.DataFrame(records["val"]),
        test=pd.DataFrame(records["test"]),
    )
    combined = []
    counts: Dict[str, Dict[str, int]] = {}
    for split_name, frame in materialized_splits.items():
        tagged = frame.copy()
        tagged.insert(0, "split", split_name)
        combined.append(tagged)
        counts[split_name] = {
            str(name): int(value)
            for name, value in frame[CLASS_COLUMN].value_counts().sort_index().items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    pd.concat(combined, ignore_index=True).to_csv(manifest_path, index=False)
    metadata_path = output_dir / METADATA_FILENAME
    metadata = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "preprocessing": asdict(config),
        "split": asdict(split_config),
        "counts": counts,
        "background_value": 0,
        "training_augmentation": "rotation_only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return LungOnlyDatasetResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        splits=materialized_splits,
        counts=counts,
    )


def load_materialized_splits(output_dir: Path | str) -> Splits:
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / MANIFEST_FILENAME, keep_default_na=False)
    required = {"split", CLASS_COLUMN, IMAGE_PATH_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Materialized manifest is missing columns: {sorted(missing)}")
    if MASK_PATH_COLUMN not in frame:
        frame[MASK_PATH_COLUMN] = ""
    return Splits(
        **{
            name: frame.loc[
                frame["split"] == name,
                [CLASS_COLUMN, IMAGE_PATH_COLUMN, MASK_PATH_COLUMN],
            ].reset_index(drop=True)
            for name in ("train", "val", "test")
        }
    )


def run_lung_only_training(
    source_dir: Path | str = PROCESSED_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    class_folders: Mapping[str, str] = CLASS_FOLDERS,
    split_config: SplitConfig = SplitConfig(),
    preprocess_config: LungOnlyPreprocessConfig = LungOnlyPreprocessConfig(),
    training_config: Optional[TransferConfig] = None,
    reports_dir: Path | str = REPORTS_DIR / "transfer_learning",
    models_dir: Path | str = MODELS_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    prepare: bool = True,
    save: bool = True,
    resume: bool = False,
    fine_tune_from_best: bool = False,
    verbose: int = 2,
) -> Tuple[LungOnlyDatasetResult, TransferResult]:
    if prepare:
        prepared = materialize_lung_only_dataset(
            source_dir, output_dir, class_folders, split_config, preprocess_config
        )
    else:
        splits = load_materialized_splits(output_dir)
        metadata_path = Path(output_dir) / METADATA_FILENAME
        prepared = LungOnlyDatasetResult(
            output_dir=Path(output_dir),
            manifest_path=Path(output_dir) / MANIFEST_FILENAME,
            metadata_path=metadata_path,
            splits=splits,
            counts={
                name: {
                    str(label): int(count)
                    for label, count in frame[CLASS_COLUMN]
                    .value_counts()
                    .sort_index()
                    .items()
                }
                for name, frame in splits.items()
            },
        )

    config = training_config or TransferConfig(
        backbone="efficientnetb0",
        image_size=preprocess_config.image_size,
        augment=False,
        horizontal_flip=False,
        balance_classes=False,
        use_class_weight=True,
        mask_lungs=False,
    )
    if config.backbone != "efficientnetb0":
        raise ValueError("The materialized lung-only process requires EfficientNetB0")
    if config.image_size != preprocess_config.image_size:
        raise ValueError("training and materialized preprocessing image sizes must match")
    if config.augment or config.balance_classes or config.mask_lungs:
        raise ValueError(
            "Images are already masked, rotated, and balanced; disable in-memory transforms"
        )
    if fine_tune_from_best and not config.fine_tune:
        raise ValueError("fine_tune_from_best requires fine_tune=True")

    checkpoint_path = None
    skip_phase1 = False
    if fine_tune_from_best:
        history_path = Path(reports_dir) / f"{model_name}_history.json"
        checkpoint_path = best_checkpoint_path(
            checkpoint_dir_for(models_dir, model_name),
            history_path,
        )
        skip_phase1 = True

    trained = run_transfer_learning(
        config=config,
        reports_dir=reports_dir,
        models_dir=models_dir,
        model_name=model_name,
        save=save,
        resume=resume,
        checkpoint_path=checkpoint_path,
        skip_phase1=skip_phase1,
        verbose=verbose,
        prepared_splits=prepared.splits,
    )
    return prepared, trained
