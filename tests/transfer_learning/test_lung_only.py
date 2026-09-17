from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from covid_xray.config import CLASS_COLUMN, IMAGE_PATH_COLUMN
from covid_xray.preprocessing import SplitConfig
from covid_xray.transfer_learning.lung_only import (
    LungOnlyPreprocessConfig,
    load_materialized_splits,
    materialize_lung_only_dataset,
)
from tl_helpers import CLASS_FOLDERS, add_synthetic_masks, build_processed_dataset, write_image


def _imbalanced_source(root: Path) -> Path:
    build_processed_dataset(root)
    add_synthetic_masks(root)
    # Add ten extra Normal examples so balancing has actual work to do.
    normal_images = root / "Normal" / "images"
    normal_masks = root / "Normal" / "masks"
    sample_image = cv2.imread(str(next(normal_images.glob("*.png"))), cv2.IMREAD_GRAYSCALE)
    sample_mask = cv2.imread(str(next(normal_masks.glob("*.png"))), cv2.IMREAD_GRAYSCALE)
    for index in range(20, 30):
        name = f"Normal-{index}.png"
        write_image(normal_images / name, sample_image)
        write_image(normal_masks / name, sample_mask)
    return root


def test_materialized_dataset_has_zero_background_and_balanced_train(tmp_path: Path) -> None:
    source = _imbalanced_source(tmp_path / "source")
    output = tmp_path / "lung_only"

    result = materialize_lung_only_dataset(
        source_dir=source,
        output_dir=output,
        class_folders=CLASS_FOLDERS,
        split_config=SplitConfig(random_state=7),
        config=LungOnlyPreprocessConfig(
            image_size=(32, 32), rotation_degrees=10, random_state=7
        ),
    )

    train_counts = result.splits.train[CLASS_COLUMN].value_counts()
    assert train_counts.nunique() == 1
    assert result.manifest_path.exists()
    assert result.metadata_path.exists()
    assert any("__rot_" in Path(path).name for path in result.splits.train[IMAGE_PATH_COLUMN])

    for split_name, frame in result.splits.items():
        assert len(frame) > 0, split_name
        for image_path in frame[IMAGE_PATH_COLUMN].head(5):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            assert image.shape == (32, 32)
            assert np.all(image[:4, :4] == 0)


def test_load_materialized_splits_round_trips_manifest(tmp_path: Path) -> None:
    source = _imbalanced_source(tmp_path / "source")
    result = materialize_lung_only_dataset(
        source,
        tmp_path / "lung_only",
        CLASS_FOLDERS,
        SplitConfig(random_state=11),
        LungOnlyPreprocessConfig(image_size=(32, 32), random_state=11),
    )

    loaded = load_materialized_splits(result.output_dir)

    for split_name, frame in result.splits.items():
        assert len(loaded[split_name]) == len(frame)
