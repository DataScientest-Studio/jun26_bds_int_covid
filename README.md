# jun26_bds_int_covid

Image classification study on the COVID-19 Radiography Database. The goal of the project is to train and evaluate a model that separates COVID-19 chest X-rays from Normal, Viral Pneumonia, and Lung Opacity images. This is an educational decision-support study, not a clinically validated diagnostic system.

The work is organized as a pipeline of steps, each one a package under `src/covid_xray/`:

| Step | Package | Status |
| --- | --- | --- |
| Preprocessing | `covid_xray.preprocessing` | Implemented |
| Training | `covid_xray.training` | Baseline implemented |
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

## Tests

```bash
pytest
```
