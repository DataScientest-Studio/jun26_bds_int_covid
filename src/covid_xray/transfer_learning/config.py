from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import DEFAULT_IMAGE_SIZE, RANDOM_STATE

BACKBONES: Tuple[str, ...] = ("efficientnetb0", "efficientnetb4", "resnet50")
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
    fine_tune_unfreeze_layers: int = 0
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-7
    augment: bool = False
    horizontal_flip: bool = True
    balance_classes: bool = False
    use_class_weight: bool = True
    mask_lungs: bool = False
    mask_only: bool = False
    crop_lungs: bool = False
    masked_pooling: bool = False
    crop_margin_fraction: float = 0.08
    random_translation: float = 0.1
    random_zoom: float = 0.1
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
        mask_modes = int(self.mask_lungs) + int(self.mask_only) + int(self.crop_lungs)
        if mask_modes > 1:
            raise ValueError("mask_lungs, mask_only, and crop_lungs are mutually exclusive")
        if self.masked_pooling and (self.mask_only or self.crop_lungs):
            raise ValueError(
                "masked_pooling is incompatible with mask_only and crop_lungs"
            )
        if not 0.0 <= self.crop_margin_fraction <= 0.5:
            raise ValueError(
                "crop_margin_fraction must be in [0, 0.5], got "
                f"{self.crop_margin_fraction}"
            )
        if self.random_translation < 0 or self.random_translation > 0.5:
            raise ValueError(
                "random_translation must be in [0, 0.5], got "
                f"{self.random_translation}"
            )
        if self.random_zoom < 0 or self.random_zoom > 0.5:
            raise ValueError(
                "random_zoom must be in [0, 0.5], got "
                f"{self.random_zoom}"
            )
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
        if self.fine_tune_unfreeze_layers < 0:
            raise ValueError(
                "fine_tune_unfreeze_layers must be >= 0, got "
                f"{self.fine_tune_unfreeze_layers}"
            )
