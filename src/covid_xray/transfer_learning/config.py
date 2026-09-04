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
    fine_tune: bool = False
    fine_tune_epochs: int = 10
    fine_tune_learning_rate: float = 1e-5
    fine_tune_early_stopping_patience: int = 5
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-7
    augment: bool = False
    horizontal_flip: bool = True
    use_class_weight: bool = True
    mask_lungs: bool = False
    mask_only: bool = False
    mask_threshold: int = 127
    early_stopping_patience: int = 3
    track_train_f1: bool = False
    save_checkpoints: bool = True
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.mask_lungs and self.mask_only:
            raise ValueError("mask_lungs and mask_only are mutually exclusive")
        if self.fine_tune and not self.pretrained:
            raise ValueError("fine_tune requires pretrained=True")
        if self.fine_tune and not self.freeze_backbone:
            raise ValueError("fine_tune requires freeze_backbone=True for phase 1")
        if self.fine_tune and self.fine_tune_epochs <= 0:
            raise ValueError(
                f"fine_tune_epochs must be positive, got {self.fine_tune_epochs}"
            )
        if self.fine_tune_learning_rate <= 0:
            raise ValueError(
                f"fine_tune_learning_rate must be positive, got {self.fine_tune_learning_rate}"
            )
