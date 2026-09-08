from __future__ import annotations

from .callbacks import MacroF1Callback
from .compare import (
    load_histories,
    load_metrics,
    model_comparison,
    plot_metric_across_runs,
)
from .config import TransferConfig
from .dataset import build_datasets, oversample_to_balance
from .gradcam import (
    aggregate_lung_focus,
    aggregate_lung_focus_by_correctness,
    build_gradcam_models,
    compute_gradcam_heatmap,
    gradcam_for_image,
    lung_attention_fraction,
    mask_image_array,
    overlay_heatmap,
    save_correctness_lung_focus_report,
    save_gradcam_grid,
    save_lung_focus_report,
    summarize_lung_focus,
)
from .history import (
    load_history,
    plot_history_comparison,
    plot_training_history,
    save_history,
)
from .model import (
    IMAGE_INPUT_NAME,
    MASK_INPUT_NAME,
    MaskedGlobalAveragePooling2D,
    TRANSFER_CUSTOM_OBJECTS,
    build_transfer_model,
    prepare_for_fine_tuning,
)
from .pipeline import (
    TransferResult,
    checkpoint_dir_for,
    default_model_name,
    format_transfer_report,
    run_transfer_learning,
)

__all__ = [
    "MacroF1Callback",
    "TransferConfig",
    "build_datasets",
    "oversample_to_balance",
    "build_transfer_model",
    "prepare_for_fine_tuning",
    "MaskedGlobalAveragePooling2D",
    "TRANSFER_CUSTOM_OBJECTS",
    "IMAGE_INPUT_NAME",
    "MASK_INPUT_NAME",
    "checkpoint_dir_for",
    "default_model_name",
    "load_histories",
    "load_history",
    "load_metrics",
    "model_comparison",
    "plot_history_comparison",
    "plot_metric_across_runs",
    "plot_training_history",
    "save_history",
    "build_gradcam_models",
    "compute_gradcam_heatmap",
    "gradcam_for_image",
    "overlay_heatmap",
    "save_gradcam_grid",
    "lung_attention_fraction",
    "mask_image_array",
    "summarize_lung_focus",
    "aggregate_lung_focus",
    "aggregate_lung_focus_by_correctness",
    "save_lung_focus_report",
    "save_correctness_lung_focus_report",
    "TransferResult",
    "format_transfer_report",
    "run_transfer_learning",
]
