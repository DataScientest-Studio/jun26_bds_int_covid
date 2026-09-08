from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FAMILY_RESULTS = [
    ("Dummy\n(majority class)", 0.4829, 0.1628),
    ("Logistic\nRegression", 0.7138, 0.7138),
    ("Hist Gradient\nBoosting", 0.8601, 0.8675),
    ("LeNet-5", 0.8171, 0.8151),
    ("CNN\n(simple)", 0.8196, 0.8271),
    ("CNN\n(scratch, VGG-style)", 0.9378, 0.9470),
    ("EfficientNetB0\n(frozen)", 0.8970, 0.9070),
    ("EfficientNetB0\n(fine-tuned)", 0.9236, 0.9356),
]

REGION_CONFOUND_RESULTS = [
    ("Logistic\nRegression", 0.7138, 0.5620, 0.6603),
    ("Hist Gradient\nBoosting", 0.8675, 0.7508, 0.8551),
    ("LeNet-5", 0.8151, 0.7386, 0.7468),
    ("CNN\n(simple)", 0.8271, 0.6904, 0.7410),
    ("CNN\n(scratch)", 0.9470, 0.9037, 0.9340),
]


def plot_model_family_comparison() -> None:
    labels = [r[0] for r in MODEL_FAMILY_RESULTS]
    accuracy = [r[1] for r in MODEL_FAMILY_RESULTS]
    macro_f1 = [r[2] for r in MODEL_FAMILY_RESULTS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_acc = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#1f77b4")
    bars_f1 = ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color="#ff7f0e")

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score (test split)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Test-split performance across all model families (full image, no augmentation)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_family_comparison.png", dpi=200)
    plt.close(fig)


def plot_region_confound_comparison() -> None:
    labels = [r[0] for r in REGION_CONFOUND_RESULTS]
    full = [r[1] for r in REGION_CONFOUND_RESULTS]
    lungs = [r[2] for r in REGION_CONFOUND_RESULTS]
    background = [r[3] for r in REGION_CONFOUND_RESULTS]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_full = ax.bar(x - width, full, width, label="Full image", color="#4c72b0")
    bars_lungs = ax.bar(x, lungs, width, label="Lungs only", color="#55a868")
    bars_bg = ax.bar(x + width, background, width, label="Background only (confound)", color="#c44e52")

    for bars in (bars_full, bars_lungs, bars_bg):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Macro F1 (test split)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Region-confound protocol: macro F1 by which pixels the model can see")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "region_confound_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_model_family_comparison()
    plot_region_confound_comparison()
    print("Saved figures to", OUTPUT_DIR.resolve())
