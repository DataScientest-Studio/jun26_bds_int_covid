from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import RANDOM_STATE
from ..training.config import REGIONS

# Architectures that can be trained from random initialization.
#   scratch -> the VGG-style stack defined by `filters`
#   lenet   -> LeNet-5, as a genuine low-capacity floor
ARCHITECTURES: Tuple[str, ...] = ("scratch", "lenet", "simple")
DEFAULT_ARCHITECTURE = "scratch"

# LeNet has only two pooling stages, so its Flatten layer grows with the square
# of the input. At 224x224 that head alone is ~5.4M parameters and the model
# stops being a small one; 32x32 is the resolution it was designed for.
DEFAULT_SCRATCH_IMAGE_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_LENET_IMAGE_SIZE: Tuple[int, int] = (32, 32)

# Above this many parameters a "LeNet" is no longer a low-capacity floor, and
# the run is more likely a resolution mistake than an intent.
LENET_PARAMETER_WARNING_THRESHOLD = 500_000

DEFAULT_REGION = "full"

# Channels per convolution block for the scratch architecture. Four blocks at
# 128x128 leaves an 8x8 feature map before global pooling.
DEFAULT_FILTERS: Tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_SIMPLE_FILTERS = (16, 32, 64)

# LeNet-5 as published uses tanh activations and average pooling. Most modern
# reimplementations silently substitute ReLU and max pooling, which is a
# different model. Both are offered here so the choice is recorded in the
# config rather than assumed by the reader.
LENET_VARIANTS: Tuple[str, ...] = ("original", "modern")
DEFAULT_LENET_VARIANT = "original"


@dataclass(frozen=True)
class CNNConfig:
    """Settings for the models trained from random initialization.

    `region` uses the same vocabulary as BaselineConfig, so any architecture
    here can be run under the identical full / lungs / background protocol as
    the sklearn baselines, and the metrics files drop straight into
    `training.compare.region_comparison`.
    """

    image_size: Tuple[int, int] = DEFAULT_SCRATCH_IMAGE_SIZE
    architecture: str = DEFAULT_ARCHITECTURE
    region: str = DEFAULT_REGION
    mask_threshold: int = 127  # masks are 0/255 PNGs
    filters: Tuple[int, ...] = DEFAULT_FILTERS
    lenet_variant: str = DEFAULT_LENET_VARIANT
    dropout_rate: float = 0.3
    spatial_dropout_rate: float = 0.1
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    augment: bool = False
    class_weight: bool = True  # mirrors class_weight="balanced" in the baselines
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    random_state: int = RANDOM_STATE
    simple_filters: Tuple[int, ...] = DEFAULT_SIMPLE_FILTERS

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}"
            )
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}")
        if self.lenet_variant not in LENET_VARIANTS:
            raise ValueError(
                f"lenet_variant must be one of {LENET_VARIANTS}, got {self.lenet_variant!r}"
            )
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if not self.filters:
            raise ValueError("filters must not be empty")


def default_image_size(architecture: str) -> Tuple[int, int]:
    """The input size each architecture is intended to run at."""
    return DEFAULT_LENET_IMAGE_SIZE if architecture == "lenet" else DEFAULT_SCRATCH_IMAGE_SIZE
