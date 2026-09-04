from __future__ import annotations

from pathlib import Path

from tl_helpers import CLASS_FOLDERS

from covid_xray.preprocessing import SplitConfig
from covid_xray.transfer_learning import TransferConfig, run_transfer_learning
from covid_xray.transfer_learning.compare import (
    load_histories,
    load_metrics,
    model_comparison,
    plot_metric_across_runs,
    plot_test_comparison,
)

SMALL = TransferConfig(
    image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8
)


def train_two_variants(processed_dir: Path, tmp_path: Path):
    reports_dir = tmp_path / "reports" / "transfer_learning"
    models_dir = tmp_path / "models"
    for name in ("variant_a", "variant_b"):
        run_transfer_learning(
            processed_dir=processed_dir,
            class_folders=CLASS_FOLDERS,
            config=SMALL,
            reports_dir=reports_dir,
            models_dir=models_dir,
            model_name=name,
            verbose=0,
        )
    return reports_dir


def test_load_metrics_collects_every_run(processed_dir: Path, tmp_path: Path) -> None:
    reports_dir = train_two_variants(processed_dir, tmp_path)

    frame = load_metrics(reports_dir)

    assert set(frame["model"]) == {"variant_a", "variant_b"}
    assert set(frame["split"]) == {"train", "val", "test"}
    assert "macro_f1" in frame.columns


def test_model_comparison_pivots_to_one_row_per_model(
    processed_dir: Path, tmp_path: Path
) -> None:
    reports_dir = train_two_variants(processed_dir, tmp_path)

    frame = model_comparison(reports_dir, split="test")

    assert list(frame.index) == ["variant_a", "variant_b"]
    assert "macro_f1" in frame.columns


def test_load_histories_returns_saved_runs(processed_dir: Path, tmp_path: Path) -> None:
    reports_dir = train_two_variants(processed_dir, tmp_path)

    histories = load_histories(["variant_a", "variant_b", "missing"], reports_dir)

    assert set(histories) == {"variant_a", "variant_b"}
    assert "val_macro_f1" in histories["variant_a"]


def test_plot_metric_across_runs_writes_a_file(processed_dir: Path, tmp_path: Path) -> None:
    reports_dir = train_two_variants(processed_dir, tmp_path)

    output_path = plot_metric_across_runs(
        ["variant_a", "variant_b"], "val_loss", tmp_path / "comparison.png", reports_dir
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_test_comparison_writes_a_file_in_label_order(
    processed_dir: Path, tmp_path: Path
) -> None:
    reports_dir = train_two_variants(processed_dir, tmp_path)
    labels = {"variant_b": "B", "variant_a": "A"}

    output_path = plot_test_comparison(labels, tmp_path / "test_comparison.png", reports_dir)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
