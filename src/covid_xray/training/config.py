from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import RANDOM_STATE

DEFAULT_BASELINE_IMAGE_SIZE: Tuple[int, int] = (64, 64)

# Which part of the X-ray the features are read from.
#   full       -> the whole image, no mask applied (the original baseline)
#   lungs      -> keep the lung field, zero out everything else
#   background -> keep everything EXCEPT the lung field (the confound control)
REGIONS: Tuple[str, ...] = ("full", "lungs", "background")
DEFAULT_REGION = "full"

@dataclass(frozen=True)
class BaselineConfig:
    image_size: Tuple[int, int] = DEFAULT_BASELINE_IMAGE_SIZE
    random_state: int = RANDOM_STATE
    logreg_max_iter: int = 3000
    logreg_C: float = 1.0
    class_weight: str = "balanced"
    hgb_max_iter: int = 200          # number of boosting rounds (trees added)
    hgb_learning_rate: float = 0.1   # how much each tree corrects the previous ones
    hgb_max_leaf_nodes: int = 31     # tree size; small = weak learner, as boosting expects
    region: str = DEFAULT_REGION
    mask_threshold: int = 127          # masks are 0/255 PNGs

    def __post_init__(self) -> None:
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}")