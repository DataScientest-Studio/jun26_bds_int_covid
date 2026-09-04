from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cnn_helpers import CLASS_FOLDERS, SMALL_CONFIG

from covid_xray.cnn import format_cnn_report, run_cnn
from covid_xray.preprocessing.config import SplitConfig

LENET_SMALL = replace(SMALL_CONFIG, architecture="lenet")


def run(raw_dir: Path, tmp_path: Path, region: str = "full", save: bool = True):
    return run_cnn(
        raw_dir=raw_dir,
        class_folders=CLASS_FOLDERS,
        split_config=SplitConfig(val_size=0.2, test_size=0.2),
        config=replace(LENET_SMALL, region=region),
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        save=save,
        verbose=0,
    )


def test_lenet_artifacts_use_the_architecture_name(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "background")

    assert result.model_path == tmp_path / "models" / "lenet_background.keras"
    assert (tmp_path / "reports" / "lenet_background_metrics.json").exists()


def test_lenet_does_not_overwrite_scratch_artifacts(raw_dir: Path, tmp_path: Path) -> None:
    scratch = run_cnn(
        raw_dir=raw_dir,
        class_folders=CLASS_FOLDERS,
        split_config=SplitConfig(val_size=0.2, test_size=0.2),
        config=SMALL_CONFIG,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        verbose=0,
    )
    lenet = run(raw_dir, tmp_path, "full")

    assert scratch.model_path != lenet.model_path
    assert scratch.model_path.exists() and lenet.model_path.exists()


def test_comparison_helper_sees_both_architectures(raw_dir: Path, tmp_path: Path) -> None:
    from covid_xray.training.compare import load_metrics

    run(raw_dir, tmp_path, "full")
    run(raw_dir, tmp_path, "background")

    frame = load_metrics(tmp_path / "reports")
    test_rows = frame[frame["split"] == "test"]

    assert set(test_rows["model"]) == {"lenet"}
    assert set(test_rows["region"]) == {"full", "background"}


def test_report_records_the_architecture(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "lungs", save=False)

    assert "architecture: lenet" in format_cnn_report(result)
