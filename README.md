# jun26_bds_int_covid

Image classification study on the COVID-19 Radiography Database. The goal of the project is to train and evaluate a model that separates COVID-19 chest X-rays from Normal, Viral Pneumonia, and Lung Opacity images. This is an educational decision-support study, not a clinically validated diagnostic system.

The work is organized as a pipeline of steps, each one a package under `src/covid_xray/`:

| Step | Package | Status |
| --- | --- | --- |
| Preprocessing | `covid_xray.preprocessing` | Implemented |
| Training | `covid_xray.training` | Baseline implemented |
| Transfer learning | `covid_xray.transfer_learning` | EfficientNetB0, frozen backbone, with Grad-CAM interpretability |
| Evaluation | `covid_xray.evaluation` | Planned |

Every step reads its inputs from disk and writes versioned artifacts, so steps can be rerun independently.

## Setup

Python 3.10+ is recommended. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The editable install puts `covid_xray` on the path for notebooks, tests, and the command line.

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

To run the notebooks, start Jupyter and select the `.venv` kernel:

```bash
jupyter notebook
```

## Dataset setup

The raw dataset is not included in this repository. Files under `data/raw/` are listed in `.gitignore` so large image data stays out of Git.

After cloning, download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle and copy the extracted contents into `data/raw/`. You should end up with class folders such as `COVID`, `Normal`, `Lung_Opacity`, and `Viral Pneumonia` (along with the metadata files) directly inside `data/raw/`.

If you use the [Kaggle API](https://www.kaggle.com/docs/api):

```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip -d data/raw
```

Processed outputs are written to `data/processed/`, which is also gitignored.

## Project layout

- `src/covid_xray/config.py`: settings shared by every step (paths, class names, label IDs, default image size, random seed).
- `src/covid_xray/preprocessing/`: the preprocessing step (details below).
- `notebooks/`: exploration and reporting; the reusable logic lives in `src/`.
- `data/raw/`, `data/processed/`, `data/arrays/`: inputs and generated artifacts, all gitignored.
- `models/`, `reports/`: trained model artifacts and written reports.
- `tests/`: mirrors the package layout, one folder per step.

## Preprocessing step

`covid_xray.preprocessing` turns the raw Kaggle download into model-ready arrays:

- `config.py`: `SplitConfig` and `PreprocessConfig` for this step.
- `files.py`: redundancy filtering and copying raw images and masks into `data/processed/`.
- `manifest.py`: manifest building and stratified train/val/test splitting.
- `transforms.py`: grayscale loading, resizing, CLAHE, lung masking, and min-max normalization.
- `augmentation.py`: seedable flip, rotation, zoom, translation, and brightness/contrast augmentation.
- `dataset.py`: array building plus `.npy` saving and loading.
- `validation.py`: comparison of a fresh build against previously saved arrays.
- `pipeline.py`: `run_preprocessing`, which runs the whole step and records its settings.

Run it from the command line:

```bash
covid-xray-preprocess --redundant-csv notebooks/redundant_images.csv
covid-xray-preprocess --skip-copy --clahe --augment --dry-run
```

Or from Python:

```python
from covid_xray.preprocessing import PreprocessConfig, format_report, run_preprocessing

result = run_preprocessing(preprocess_config=PreprocessConfig(apply_clahe=True))
print(format_report(result))
```

Outputs go to `data/arrays/`: `X_{split}.npy`, `y_{split}.npy`, and `preprocessing.json`, which records the seed, split ratios, preprocessing and augmentation settings, per-class counts, and package versions so the training step knows exactly what it is consuming. Before saving, the step compares the fresh build against the arrays already on disk and reports any change in sample counts, labels, or pixel values.

## Training step: raw-pixel baseline

`covid_xray.training` implements the simplest possible baseline models, trained directly on raw pixels rather than on the engineered `data/arrays/` outputs, so later models have a floor to beat:

- `config.py`: `BaselineConfig` (image size for downsampling, seed, logistic regression settings).
- `data.py`: builds a manifest straight from `data/raw/` and drops known redundant files.
- `features.py`: reads each raw image in grayscale, downsamples it (64x64 by default), flattens it, and scales pixel values to `[0, 1]`. No CLAHE, lung masking, or augmentation is applied, since the goal is a raw-pixel baseline.
- `models.py`: a majority-class `DummyClassifier` (the trivial floor) and a `LogisticRegression` with `class_weight="balanced"` (a simple linear baseline that still accounts for class imbalance).
- `evaluation.py`: per-class precision/recall/F1 (via `classification_report`), confusion matrices, and plots.
- `pipeline.py`: `run_baseline`, which builds its own stratified 70/15/15 split (same ratios and seed as the preprocessing step) and fits/evaluates both models on train and test.

Run it from the command line:

```bash
covid-xray-train-baseline --redundant-csv data/metadata/redundant_images.csv
```

Or from Python:

```python
from covid_xray.training import format_baseline_report, run_baseline

result = run_baseline(redundant_csv="data/metadata/redundant_images.csv")
print(format_baseline_report(result))
```

Outputs go to `models/baseline_dummy.joblib`, `models/baseline_logistic_regression.joblib`, and `reports/baseline/` (per-model metrics JSON and confusion matrix plots for train and test).

## Training step: transfer learning (EfficientNetB0 / ResNet50)

`covid_xray.transfer_learning` fine-tunes an ImageNet-pretrained backbone (Keras/TensorFlow) on chest X-rays. Two backbones are supported via `TransferConfig(backbone=...)`: `"efficientnetb0"` (default) and `"resnet50"`. Unlike the baseline, it reads images directly from `data/processed/` (full resolution PNGs), not the `data/arrays/` `.npy` files, and keeps pixel values in `[0, 255]`: EfficientNet's Keras implementation normalizes internally, and for ResNet50 the model graph applies `keras.applications.resnet50.preprocess_input` automatically before the backbone.

By default the ImageNet backbone is **fully frozen** and only a small classification head (`GlobalAveragePooling -> Dropout -> Dense -> Dropout -> Dense(softmax)`) is trained on top of it. This "feature extraction" approach is the safest starting point for transfer learning: it trains fast, needs little data, and is unlikely to overfit or destroy the pretrained features. Fine-tuning (unfreezing some backbone layers for a low-learning-rate second pass) can be layered on later via `TransferConfig(freeze_backbone=False)` or `--fine-tune`.

- `config.py`: `TransferConfig` (backbone, image size, batch size, epochs, learning rate, dropout, `pretrained` to toggle ImageNet weights, `augment` for light flip/rotation augmentation, `balance_classes` to equalize class counts).
- `dataset.py`: builds `tf.data.Dataset` pipelines straight from the manifest — decode PNG, resize, convert grayscale to 3-channel, batch, prefetch. Also has `oversample_to_balance`, which duplicates minority-class rows up to the largest class's count for exact class balance.
- `model.py`: `build_transfer_model`, wiring the frozen (or unfrozen) backbone (EfficientNetB0 or ResNet50) to the new head, including any backbone-specific input preprocessing.
- `evaluation.py`: batched prediction + the same `EvaluationResult`/metrics/confusion-matrix format as the baseline, so results are directly comparable.
- `pipeline.py`: `run_transfer_learning`, using the same manifest, same 70/15/15 stratified split, and same seed (42) as the other steps. Saved model/report filenames default to `transfer_<backbone>` when `--model-name`/`model_name` is omitted.

Run it from the command line (defaults to EfficientNetB0):

```bash
covid-xray-train-transfer --epochs 10 --batch-size 32
```

Or select ResNet50:

```bash
covid-xray-train-transfer --backbone resnet50 --epochs 10 --batch-size 32
```

Or from Python:

```python
from covid_xray.transfer_learning import TransferConfig, format_transfer_report, run_transfer_learning

result = run_transfer_learning(config=TransferConfig(backbone="resnet50", epochs=10))
print(format_transfer_report(result))
```

Outputs go to `models/transfer_efficientnetb0.keras` (or `models/transfer_resnet50.keras`) and `reports/transfer_learning/` (metrics JSON and confusion matrix plots for train/val/test).

### Balancing classes: equal case counts via oversampling + augmentation

By default the dataset is imbalanced (Normal and Lung_Opacity outnumber COVID and Viral Pneumonia). Two strategies address this, and they can be combined with `mask_lungs` to also remove all non-lung background pixels:

- `use_class_weight` (`--class-weight`): keeps every real sample, but weights the loss so minority classes count more. The training set stays imbalanced in raw counts.
- `balance_classes` (`--balance-classes`): actually equalizes the training set's class counts by duplicating minority-class rows up to the size of the largest class (`oversample_to_balance`). Every duplicated row is passed through the augmentation pipeline (rotation, plus horizontal flip unless disabled) so it is not a pixel-identical copy of its source image; original rows are left untouched unless `--augment` is also set. Validation and test splits are never touched, so evaluation stays on the true class distribution.

Since chest X-ray anatomy is not left/right symmetric in a way that should be flipped, pair `--balance-classes` with `--no-horizontal-flip` to only use rotation for the synthetic copies:

```bash
covid-xray-train-transfer --balance-classes --no-horizontal-flip --mask-lungs
```

This trains on an exactly class-balanced set, with the background fully zeroed out via the lung segmentation mask, using rotation-only augmentation (no flip) to fill in the extra copies needed for the minority classes.

### Interpretability: Grad-CAM and lung-focus analysis

`covid_xray.transfer_learning.gradcam` answers a key trust question: does the model actually learn from the lungs, or is it partly relying on background/border cues? It generates Grad-CAM visualizations, quantifies what share of the model's attention falls inside the lung segmentation mask versus a chance baseline, and compares a normal model against one trained with the background forcibly masked out (`TransferConfig(mask_lungs=True)` / `--mask-lungs`).

See **[`reports/transfer_learning/README.md`](reports/transfer_learning/README.md)** for the full write-up, including a finding that masking the background dropped COVID recall from 90% to 57% — direct evidence that a meaningful share of the model's COVID accuracy came from non-anatomical shortcuts in the dataset rather than lung pathology.

```bash
python -m covid_xray.transfer_learning.gradcam_cli \
  --model-path models/transfer_efficientnetb0.keras \
  --output reports/transfer_learning/gradcam.png \
  --samples-per-class 2 --lung-focus-sample-size 300
```

Notes:

- The first run downloads ImageNet weights (~16 MB) and caches them under `~/.keras/`; it needs internet access once.
- If macOS raises an SSL certificate error while downloading the weights (a known issue with the python.org installer), run `pip install certifi` and set `SSL_CERT_FILE` to `python -c "import certifi; print(certifi.where())"`, or run the "Install Certificates.command" script bundled with your Python installation.
- Training runs on CPU but is far faster on a GPU; use `--batch-size` and `--epochs` to size the run to your hardware.

## Tests

```bash
pytest
```
