"""Parse a Keras verbose=2 training log and plot the learning curves.

Usage:
    python plot_training_log.py reports/cnn/train_log_background.txt
    python plot_training_log.py reports/cnn/train_log.txt --out curves.png

Works on any log written with `verbose=2`, which prints one line per epoch:

    924/924 - 155s - 167ms/step - accuracy: 0.6761 - loss: 0.6920 - ...

Metrics are discovered from the line rather than hardcoded, so the same script
handles the baseline, CNN, and transfer-learning logs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed; write straight to a file
import matplotlib.pyplot as plt

# "accuracy: 0.6761", "val_loss: 0.9770", "learning_rate: 0.0010"
# Excludes timing fields (155s, 167ms/step) because those have no colon.
METRIC_PATTERN = re.compile(r"([a-z_]+):\s*([0-9.eE+-]+)")

# An epoch line always starts with "<batches>/<batches> - <seconds>s".
EPOCH_LINE_PATTERN = re.compile(r"^\s*\d+/\d+\s+-\s+\d+")


def parse_log(path: Path) -> dict[str, list[float]]:
    """Pull per-epoch metrics out of the log, in order."""
    history: dict[str, list[float]] = {}

    for line in path.read_text(errors="replace").splitlines():
        if not EPOCH_LINE_PATTERN.match(line):
            continue
        for name, value in METRIC_PATTERN.findall(line):
            try:
                history.setdefault(name, []).append(float(value))
            except ValueError:
                continue

    if not history:
        raise ValueError(
            f"No epoch lines found in {path}. "
            "Was the run started with verbose=2?"
        )
    return history


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> Path:
    """One panel per metric family: loss, accuracy, learning rate."""
    families = [
        ("loss", [k for k in history if k.endswith("loss")]),
        ("accuracy", [k for k in history if k.endswith("accuracy")]),
        ("learning rate", [k for k in history if "learning_rate" in k]),
    ]
    families = [(label, keys) for label, keys in families if keys]

    figure, axes = plt.subplots(
        1, len(families), figsize=(5 * len(families), 4), squeeze=False
    )

    for axis, (label, keys) in zip(axes[0], families):
        for key in sorted(keys):
            epochs = range(1, len(history[key]) + 1)
            axis.plot(epochs, history[key], marker="o", markersize=3, label=key)
        axis.set_xlabel("epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

        # Mark the best epoch: this is the checkpoint EarlyStopping restores,
        # so it is the run that the reported metrics actually come from.
        if label == "loss" and "val_loss" in history:
            best = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
            axis.axvline(best + 1, color="crimson", linestyle="--", linewidth=1)
            axis.set_title(f"best val_loss @ epoch {best + 1}")
        else:
            axis.set_title(label)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def summarize(history: dict[str, list[float]]) -> str:
    lines = [f"Epochs parsed: {len(next(iter(history.values())))}"]

    if "val_loss" in history:
        val_loss = history["val_loss"]
        best = min(range(len(val_loss)), key=val_loss.__getitem__)
        lines.append(f"Best val_loss: {val_loss[best]:.4f} at epoch {best + 1}")
        # Epochs since the last improvement -- compare against EarlyStopping
        # patience to see whether the run was about to stop on its own.
        lines.append(f"Epochs since improvement: {len(val_loss) - best - 1}")

    if "loss" in history and "val_loss" in history:
        gap = history["val_loss"][-1] - history["loss"][-1]
        lines.append(f"Final val_loss - loss: {gap:+.4f}")
        lines.append(
            "  (large positive = overfitting; near zero or negative = underfitting)"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    history = parse_log(args.log)
    output_path = args.out or args.log.with_suffix(".png")

    print(summarize(history))
    plot_history(history, output_path, args.log.stem)
    print(f"\nSaved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
