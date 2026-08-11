from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import RANDOM_STATE

DEFAULT_BASELINE_IMAGE_SIZE: Tuple[int, int] = (64, 64)


@dataclass(frozen=True)
class BaselineConfig:
    image_size: Tuple[int, int] = DEFAULT_BASELINE_IMAGE_SIZE
    random_state: int = RANDOM_STATE
    logreg_max_iter: int = 3000
    logreg_C: float = 1.0
    class_weight: str = "balanced"
