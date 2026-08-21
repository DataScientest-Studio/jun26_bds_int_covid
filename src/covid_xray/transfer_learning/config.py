from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import DEFAULT_IMAGE_SIZE, RANDOM_STATE

BACKBONES: Tuple[str, ...] = ("efficientnetb0",)
DEFAULT_BACKBONE = "efficientnetb0"


@dataclass(frozen=True)
class TransferConfig:
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    backbone: str = DEFAULT_BACKBONE
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    dense_units: int = 128
    dropout_rate: float = 0.3
    freeze_backbone: bool = True
    pretrained: bool = True
    augment: bool = False
    use_class_weight: bool = True
    mask_lungs: bool = False
    mask_threshold: int = 127
    early_stopping_patience: int = 3
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
