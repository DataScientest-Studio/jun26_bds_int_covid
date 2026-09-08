from __future__ import annotations

from pathlib import Path

import pytest
from tl_helpers import CLASS_FOLDERS, IMAGES_PER_CLASS

from covid_xray.preprocessing import SplitConfig
from covid_xray.transfer_learning import (
    TransferConfig,
    default_model_name,
    format_transfer_report,
    run_transfer_learning,
)
from covid_xray.transfer_learning.__main__ import main

SMALL = TransferConfig(
    image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8
)
RESNET_SMALL = TransferConfig(
    image_size=(64, 64),
    backbone="resnet50",
    pretrained=False,
    batch_size=4,
    epochs=1,
    dense_units=8,
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


def test_run_transfer_learning_tracks_val_macro_f1_per_epoch(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=2, dense_units=8,
        early_stopping_patience=99,
    )
    result = run_step(processed_dir, tmp_path, config=config)

    assert "val_macro_f1" in result.history
    assert len(result.history["val_macro_f1"]) == len(result.history["loss"])


def test_run_transfer_learning_saves_history_json_and_plot(
    processed_dir: Path, tmp_path: Path
) -> None:
    result = run_step(processed_dir, tmp_path)

    assert result.history_path is not None
    assert result.history_path.exists()
    assert (
        tmp_path / "reports" / "transfer_learning" / "test_transfer_history.png"
    ).exists()


def test_run_transfer_learning_saves_checkpoint_per_epoch(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=2, dense_units=8,
        early_stopping_patience=99,
    )
    run_step(processed_dir, tmp_path, config=config)

    checkpoint_dir = tmp_path / "models" / "checkpoints" / "test_transfer"
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.keras"))
    assert [path.name for path in checkpoints] == ["epoch_0001.keras", "epoch_0002.keras"]


def test_run_transfer_learning_can_disable_checkpoints(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        save_checkpoints=False,
    )
    run_step(processed_dir, tmp_path, config=config)

    checkpoint_dir = tmp_path / "models" / "checkpoints" / "test_transfer"
    assert not checkpoint_dir.exists()


def test_run_transfer_learning_resumes_from_latest_checkpoint(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        early_stopping_patience=99,
    )
    run_step(processed_dir, tmp_path, config=config)

    resumed_config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=2, dense_units=8,
        early_stopping_patience=99,
    )
    result = run_step(processed_dir, tmp_path, config=resumed_config, resume=True)

    checkpoint_dir = tmp_path / "models" / "checkpoints" / "test_transfer"
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.keras"))
    assert [path.name for path in checkpoints] == ["epoch_0001.keras", "epoch_0002.keras"]
    assert len(result.history["loss"]) == 2


def test_run_transfer_learning_with_class_weight_trains_and_evaluates(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        use_class_weight=True,
    )
    result = run_step(processed_dir, tmp_path, config=config)

    assert set(result.evaluations) == {"train", "val", "test"}
    assert result.model_path is not None
    assert result.model_path.exists()


def test_run_transfer_learning_with_balance_classes_equalizes_train_counts(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        balance_classes=True, horizontal_flip=False, use_class_weight=False,
    )
    result = run_step(processed_dir, tmp_path, config=config)

    train_counts = result.splits.train["class"].value_counts()
    assert train_counts.nunique() == 1
    assert set(result.evaluations) == {"train", "val", "test"}
    assert result.model_path is not None
    assert result.model_path.exists()


def test_run_transfer_learning_with_balance_classes_and_crop_lungs(
    processed_dir: Path, tmp_path: Path
) -> None:
    from tl_helpers import add_synthetic_masks

    add_synthetic_masks(processed_dir)
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        balance_classes=True, horizontal_flip=False, use_class_weight=False,
        crop_lungs=True,
    )
    result = run_step(processed_dir, tmp_path, config=config)

    train_counts = result.splits.train["class"].value_counts()
    assert train_counts.nunique() == 1
    assert result.model_path is not None
    assert result.model_path.exists()


def test_run_transfer_learning_with_balance_classes_and_mask_lungs(
    processed_dir: Path, tmp_path: Path
) -> None:
    from tl_helpers import add_synthetic_masks

    add_synthetic_masks(processed_dir)
    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=1, dense_units=8,
        balance_classes=True, horizontal_flip=False, use_class_weight=False,
        mask_lungs=True,
    )
    result = run_step(processed_dir, tmp_path, config=config)

    train_counts = result.splits.train["class"].value_counts()
    assert train_counts.nunique() == 1
    assert result.model_path is not None
    assert result.model_path.exists()


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


def test_cli_class_weight_flag_runs_end_to_end(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(cli_args(processed_dir, tmp_path) + ["--class-weight"])

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


def test_cli_balance_classes_flag_runs_end_to_end(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(
        cli_args(processed_dir, tmp_path) + ["--balance-classes", "--no-horizontal-flip"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_cli_no_horizontal_flip_flag_runs_end_to_end(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(
        cli_args(processed_dir, tmp_path) + ["--augment", "--no-horizontal-flip"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_cli_resume_flag_continues_from_checkpoint(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    main(cli_args(processed_dir, tmp_path))
    capsys.readouterr()

    exit_code = main(cli_args(processed_dir, tmp_path) + ["--epochs", "2", "--resume"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_run_transfer_learning_with_fine_tune_trains_both_phases(
    processed_dir: Path, tmp_path: Path
) -> None:
    config = TransferConfig(
        image_size=(64, 64),
        pretrained=True,
        batch_size=4,
        epochs=1,
        fine_tune=True,
        fine_tune_epochs=1,
        dense_units=8,
        early_stopping_patience=99,
        fine_tune_early_stopping_patience=99,
    )
    result = run_step(processed_dir, tmp_path, config=config, model_name="test_finetune")

    assert len(result.history["loss"]) == 2
    assert len(result.history["val_macro_f1"]) == 2
    assert result.model_path is not None
    assert result.model_path.exists()


def test_run_transfer_learning_resumes_correctly_from_interrupted_fine_tune(
    processed_dir: Path, tmp_path: Path
) -> None:
    import shutil

    config = TransferConfig(
        image_size=(64, 64), pretrained=False, batch_size=4, epochs=2, dense_units=8,
        early_stopping_patience=99,
    )
    run_step(processed_dir, tmp_path, config=config, model_name="test_finetune_resume")

    checkpoint_dir = tmp_path / "models" / "checkpoints" / "test_finetune_resume"
    shutil.copy(checkpoint_dir / "epoch_0002.keras", checkpoint_dir / "epoch_0004.keras")
    epoch_4_bytes_before = (checkpoint_dir / "epoch_0004.keras").read_bytes()

    resumed_config = TransferConfig(
        image_size=(64, 64), pretrained=True, batch_size=4, epochs=2, dense_units=8,
        fine_tune=True, fine_tune_epochs=4, fine_tune_early_stopping_patience=99,
    )
    result = run_step(
        processed_dir, tmp_path, config=resumed_config,
        model_name="test_finetune_resume", resume=True,
    )

    checkpoints = sorted(checkpoint_dir.glob("epoch_*.keras"))
    assert [path.name for path in checkpoints] == [
        "epoch_0001.keras",
        "epoch_0002.keras",
        "epoch_0004.keras",
        "epoch_0005.keras",
        "epoch_0006.keras",
    ]
    assert (checkpoint_dir / "epoch_0004.keras").read_bytes() == epoch_4_bytes_before
    assert len(result.history["loss"]) == 4


def test_cli_fine_tune_flag_runs_end_to_end(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(
        [
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
            "--fine-tune",
            "--fine-tune-epochs",
            "1",
            "--fine-tune-early-stopping-patience",
            "99",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_cli_rejects_fine_tune_with_unfreeze_backbone(
    processed_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(cli_args(processed_dir, tmp_path) + ["--fine-tune", "--unfreeze-backbone"])


def test_run_transfer_learning_supports_resnet50_backbone(
    processed_dir: Path, tmp_path: Path
) -> None:
    result = run_step(processed_dir, tmp_path, config=RESNET_SMALL, model_name="test_resnet50")

    assert set(result.evaluations) == {"train", "val", "test"}
    assert result.model_path is not None
    assert result.model_path.exists()


def test_default_model_name_is_backbone_specific() -> None:
    assert default_model_name("efficientnetb0") == "transfer_efficientnetb0"
    assert default_model_name("efficientnetb4") == "transfer_efficientnetb4"
    assert default_model_name("resnet50") == "transfer_resnet50"


def test_run_transfer_learning_defaults_model_name_to_backbone(
    processed_dir: Path, tmp_path: Path
) -> None:
    result = run_transfer_learning(
        processed_dir=processed_dir,
        class_folders=CLASS_FOLDERS,
        config=RESNET_SMALL,
        reports_dir=tmp_path / "reports" / "transfer_learning",
        models_dir=tmp_path / "models",
        verbose=0,
    )

    assert result.model_path is not None
    assert result.model_path.name == "transfer_resnet50.keras"


def test_cli_backbone_flag_runs_end_to_end_for_resnet50(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(
        cli_args(processed_dir, tmp_path) + ["--backbone", "resnet50"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved model" in output


def test_cli_defaults_model_name_from_backbone(
    processed_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    args = [
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
        "--backbone",
        "resnet50",
    ]
    exit_code = main(args)

    assert exit_code == 0
    capsys.readouterr()
    assert (tmp_path / "models" / "transfer_resnet50.keras").exists()
