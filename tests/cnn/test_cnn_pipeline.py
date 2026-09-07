from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cnn_helpers import CLASS_FOLDERS, SMALL_CONFIG

from covid_xray.cnn import format_cnn_report, run_cnn
from covid_xray.cnn.pipeline import default_model_name
from covid_xray.preprocessing.config import SplitConfig


def run(raw_dir: Path, tmp_path: Path, region: str, save: bool = True):
    return run_cnn(
        raw_dir=raw_dir,
        class_folders=CLASS_FOLDERS,
        split_config=SplitConfig(val_size=0.2, test_size=0.2),
        config=replace(SMALL_CONFIG, region=region),
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        save=save,
        verbose=0,
    )


def test_pipeline_evaluates_every_split(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "full", save=False)

    assert set(result.evaluations) == {"train", "val", "test"}


def test_pipeline_writes_region_suffixed_artifacts(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "background")

    assert result.model_path == tmp_path / "models" / f"{default_model_name('scratch')}_background.keras"
    assert result.model_path.exists()
    assert (
        tmp_path / "reports" / f"{default_model_name('scratch')}_background_metrics.json"
    ).exists()


def test_full_region_artifacts_are_unsuffixed(raw_dir: Path, tmp_path: Path) -> None:
    # Matches the baseline convention, so compare.load_metrics reads the
    # unsuffixed file as the full-image run.
    result = run(raw_dir, tmp_path, "full")

    assert result.model_path.name == f"{default_model_name('scratch')}.keras"


def test_metrics_json_is_readable_by_the_comparison_helper(
    raw_dir: Path, tmp_path: Path
) -> None:
    from covid_xray.training.compare import load_metrics

    run(raw_dir, tmp_path, "full")
    run(raw_dir, tmp_path, "background")

    frame = load_metrics(tmp_path / "reports")
    regions = set(frame[frame["split"] == "test"]["region"])

    assert regions == {"full", "background"}


def test_regions_share_split_membership(raw_dir: Path, tmp_path: Path) -> None:
    full = run(raw_dir, tmp_path, "full", save=False)
    background = run(raw_dir, tmp_path, "background", save=False)

    # If this drifts, the region comparison is measuring two different
    # test sets rather than two different pixel subsets.
    assert list(full.splits.test["image_path"]) == list(background.splits.test["image_path"])


def test_class_weights_are_disabled_when_configured(raw_dir: Path, tmp_path: Path) -> None:
    result = run_cnn(
        raw_dir=raw_dir,
        class_folders=CLASS_FOLDERS,
        split_config=SplitConfig(val_size=0.2, test_size=0.2),
        config=replace(SMALL_CONFIG, class_weight=False),
        save=False,
        verbose=0,
    )

    assert result.class_weights == {}


def test_report_mentions_region_and_parameters(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "lungs", save=False)
    text = format_cnn_report(result)

    assert "region: lungs" in text
    assert "parameters:" in text


def test_history_is_written_and_json_serializable(raw_dir: Path, tmp_path: Path) -> None:
    result = run(raw_dir, tmp_path, "full")
    path = tmp_path / "reports" / f"{default_model_name('scratch')}_history.json"

    assert path.exists()
    payload = json.loads(path.read_text())
    assert set(payload) == set(result.history)
    assert len(payload["loss"]) == len(result.history["loss"])
    assert all(isinstance(v, float) for v in payload["loss"])


def test_history_is_not_read_as_a_model_by_the_comparison_helper(
    raw_dir: Path, tmp_path: Path
) -> None:
    from covid_xray.training.compare import load_metrics

    run(raw_dir, tmp_path, "full")
    frame = load_metrics(tmp_path / "reports")

    assert set(frame["model"]) == {default_model_name("scratch")}


def test_checkpoint_is_written_alongside_the_final_model(
    raw_dir: Path, tmp_path: Path
) -> None:
    run(raw_dir, tmp_path, "lungs")
    models = tmp_path / "models"

    assert (models / f"{default_model_name('scratch')}_lungs.keras").exists()
    assert (models / f"{default_model_name('scratch')}_lungs_ckpt.keras").exists()


def test_dry_run_writes_no_checkpoint(raw_dir: Path, tmp_path: Path) -> None:
    run(raw_dir, tmp_path, "full", save=False)

    assert not (tmp_path / "models").exists()
