from __future__ import annotations

from pathlib import Path

from covid_xray.transfer_learning.history import (
    load_history,
    plot_history_comparison,
    plot_training_history,
    save_history,
)

HISTORY = {
    "loss": [0.9, 0.6, 0.4],
    "val_loss": [0.95, 0.7, 0.55],
    "accuracy": [0.6, 0.75, 0.85],
    "val_accuracy": [0.55, 0.7, 0.78],
    "val_macro_f1": [0.5, 0.68, 0.77],
}


def test_save_and_load_history_round_trips(tmp_path: Path) -> None:
    output_path = save_history(HISTORY, tmp_path / "run_history.json")
    loaded = load_history(output_path)

    assert loaded == HISTORY


def test_plot_training_history_writes_a_file(tmp_path: Path) -> None:
    output_path = plot_training_history(HISTORY, tmp_path / "history.png", model_name="demo")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_training_history_handles_missing_f1(tmp_path: Path) -> None:
    minimal_history = {"loss": [0.9, 0.5], "val_loss": [1.0, 0.6]}

    output_path = plot_training_history(minimal_history, tmp_path / "history.png")

    assert output_path.exists()


def test_plot_history_comparison_writes_a_file(tmp_path: Path) -> None:
    other_history = {"val_loss": [1.1, 0.9, 0.8]}

    output_path = plot_history_comparison(
        {"run_a": HISTORY, "run_b": other_history}, "val_loss", tmp_path / "comparison.png"
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
