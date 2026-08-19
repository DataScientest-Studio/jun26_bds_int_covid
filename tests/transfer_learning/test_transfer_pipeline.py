from __future__ import annotations

from pathlib import Path

import pytest
from tl_helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import SplitConfig
from covid_xray.transfer_learning import (
    TransferConfig,
    format_transfer_report,
    run_transfer_learning,
)
from covid_xray.transfer_learning.__main__ import main

SMALL = TransferConfig(
    image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8
)


def run_step(processed_dir: Path, tmp_path: Path, **overrides):
    defaults = dict(
        processed_dir=processed_dir,
        class_folders=CLASS_FOLDERS,
        config=SMALL,
        reports_dir=tmp_path / "reports" / "transfer_learning",
        models_dir=tmp_path / "models",
        model_name="test_transfer",
        verbose=0,
    )
    return run_transfer_learning(**{**defaults, **overrides})


def test_run_transfer_learning_trains_and_evaluates(
    processed_dir: Path, tmp_path: Path
) -> None:
    result = run_step(processed_dir, tmp_path)

    assert result.splits.total == IMAGES_PER_CLASS * len(CLASS_FOLDERS)
    assert set(result.evaluations) == {"train", "val", "test"}
    assert result.model_path is not None
    assert result.model_path.exists()
    assert (tmp_path / "reports" / "transfer_learning" / "test_transfer_metrics.json").exists()


def test_run_transfer_learning_dry_run_writes_nothing(
    processed_dir: Path, tmp_path: Path
) -> None:
    result = run_step(processed_dir, tmp_path, save=False)

    assert result.model_path is None
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports").exists()
    assert result.splits.total > 0


def test_format_transfer_report_mentions_saved_model(
    processed_dir: Path, tmp_path: Path
) -> None:
    report = format_transfer_report(run_step(processed_dir, tmp_path))

    assert "Saved model" in report


def cli_args(processed_dir: Path, tmp_path: Path) -> list:
    return [
        "--processed-dir",
        str(processed_dir),
        "--models-dir",
        str(tmp_path / "models"),
        "--reports-dir",
        str(tmp_path / "reports" / "transfer_learning"),
        "--classes",
        *CLASS_FOLDERS,
        "--image-size",
        "64",
        "64",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--dense-units",
        "8",
        "--no-pretrained",
    ]


def test_cli_runs_transfer_learning_end_to_end(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(cli_args(processed_dir, tmp_path) + ["--seed", "7"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_cli_dry_run_saves_nothing(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(cli_args(processed_dir, tmp_path) + ["--dry-run"])

    capsys.readouterr()
    assert exit_code == 0
    assert not (tmp_path / "models").exists()
