from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import SplitConfig
from covid_xray.training import BaselineConfig, format_baseline_report, run_baseline
from covid_xray.training.__main__ import main

SMALL = BaselineConfig(image_size=(16, 16))

MODEL_NAMES = ("dummy", "logistic_regression", "hist_gradient_boosting")

def run_step(raw_dir: Path, tmp_path: Path, **overrides):
    defaults = dict(
        raw_dir=raw_dir,
        class_folders=CLASS_FOLDERS,
        config=SMALL,
        reports_dir=tmp_path / "reports" / "baseline",
        models_dir=tmp_path / "models",
    )
    return run_baseline(**{**defaults, **overrides})


def test_run_baseline_trains_and_evaluates_both_models(raw_dir: Path, tmp_path: Path) -> None:
    result = run_step(raw_dir, tmp_path)

    assert result.splits.total == IMAGES_PER_CLASS * len(CLASS_FOLDERS)
    assert set(result.evaluations) == set(MODEL_NAMES)
    for split_evals in result.evaluations.values():
        assert set(split_evals) == {"train", "test"}
    assert set(result.model_paths) == set(MODEL_NAMES)
    for path in result.model_paths.values():
        assert path.exists()


def test_run_baseline_excludes_redundant_files(raw_dir: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "redundant_images.csv"
    pd.DataFrame(
        {"class": ["COVID"], "redundant_file": ["COVID-0.png"]}
    ).to_csv(csv_path, index=False)

    result = run_step(raw_dir, tmp_path, redundant_csv=csv_path)

    assert result.splits.total == IMAGES_PER_CLASS * len(CLASS_FOLDERS) - 1


def test_run_baseline_dry_run_writes_nothing(raw_dir: Path, tmp_path: Path) -> None:
    result = run_step(raw_dir, tmp_path, save=False)

    assert result.model_paths == {}
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports").exists()
    assert result.splits.total > 0


def test_dummy_baseline_only_predicts_the_majority_class(raw_dir: Path, tmp_path: Path) -> None:
    result = run_step(raw_dir, tmp_path)

    dummy_test = result.evaluations["dummy"]["test"]
    predicted_classes = {
        class_name
        for class_name in dummy_test.report
        if class_name not in {"accuracy", "macro avg", "weighted avg"}
        and dummy_test.report[class_name]["support"] > 0
        and dummy_test.report[class_name]["recall"] > 0
    }
    assert len(predicted_classes) <= 1


def test_format_baseline_report_covers_both_models(raw_dir: Path, tmp_path: Path) -> None:
    report = format_baseline_report(run_step(raw_dir, tmp_path))

    assert "dummy" in report
    assert "logistic_regression" in report
    assert "Saved models" in report


def cli_args(raw_dir: Path, tmp_path: Path) -> list:
    return [
        "--raw-dir",
        str(raw_dir),
        "--models-dir",
        str(tmp_path / "models"),
        "--reports-dir",
        str(tmp_path / "reports" / "baseline"),
        "--classes",
        *CLASS_FOLDERS,
        "--image-size",
        "16",
        "16",
    ]


def test_cli_runs_the_baseline_end_to_end(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(cli_args(raw_dir, tmp_path) + ["--seed", "7"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved models" in output
    assert (tmp_path / "models" / "baseline_dummy.joblib").exists()


def test_cli_dry_run_saves_nothing(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(cli_args(raw_dir, tmp_path) + ["--dry-run"])

    capsys.readouterr()
    assert exit_code == 0
    assert not (tmp_path / "models").exists()
