from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import CLASS_COLUMN, DEFAULT_IMAGE_SIZE, RANDOM_STATE


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = RANDOM_STATE
    stratify_column: str = CLASS_COLUMN

    def __post_init__(self) -> None:
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.val_size + self.test_size >= 1:
            raise ValueError(
                "val_size + test_size must be below 1, got "
                f"{self.val_size + self.test_size}"
            )

    @property
    def holdout_size(self) -> float:
        return self.val_size + self.test_size

    @property
    def test_size_within_holdout(self) -> float:
        return self.test_size / self.holdout_size


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    apply_lung_mask: bool = False
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    mask_threshold: int = 127
    augment: bool = False
    random_state: int = RANDOM_STATE
