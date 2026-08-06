# Preprocessing step

This package is one step in the COVID X-ray classification project. It takes the raw Kaggle dataset and produces model-ready NumPy arrays on disk. Training is a separate step and will read those arrays (or, later, a manifest with lazy loading).

The entry point is `run_preprocessing()` in `pipeline.py`. Everything else is a helper called from there or exposed for notebooks and tests.

## What goes in and what comes out

```
data/raw/                          data/processed/              data/arrays/
  COVID/images/*.png                 COVID/images/*.png           X_train.npy
  COVID/masks/*.png        -->       COVID/masks/*.png    -->     y_train.npy
  Normal/...                         Normal/...                   X_val.npy, y_val.npy
  Lung_Opacity/...                   ...                          X_test.npy, y_test.npy
  Viral Pneumonia/...                                             preprocessing.json
```

| Location | Role |
| --- | --- |
| `data/raw/` | Original download from Kaggle. Never modified by this step. |
| `data/processed/` | Clean copy of images and masks after dropping redundant files. Same folder layout as raw. |
| `data/arrays/` | Final tensors: one feature array and one label array per split, plus metadata JSON. |

Shared paths and class names live in `../config.py`. Settings that only affect preprocessing live in `config.py` inside this folder.

## Pipeline order

`run_preprocessing()` runs these stages in sequence:

```
1. copy (optional)     files.py
2. manifest            manifest.py
3. split               manifest.py
4. preprocess + stack  transforms.py, augmentation.py, dataset.py
5. count check         validation.py
6. compare to disk     validation.py
7. save (optional)     dataset.py, pipeline.py
```

Each stage passes concrete objects to the next (paths, DataFrames, NumPy arrays). Nothing is global state except the default paths in `../config.py`.

### 1. Copy and filter (`files.py`)

**Input:** `data/raw/{Class}/images/*.png` and matching masks.

**What happens:**

1. If `redundant_csv` is given, read `redundant_images.csv`. Each row is `(class, filename)` to exclude (duplicate or bad images found in exploration).
2. For each class folder, walk `images/` in sorted filename order.
3. Skip files listed in the redundancy set.
4. Copy the PNG to `data/processed/{Class}/images/` with `shutil.copy2` (preserves metadata).
5. Copy the mask with the same basename from `masks/` if it exists; count missing masks but still copy the image.

**Output:** Files on disk under `data/processed/`. A `CopyResult` per class with counts: copied, skipped, masks copied, masks missing.

The raw folder is never touched. Re-running copy overwrites processed files with the same names.

### 2. Build manifest (`manifest.py`)

**Input:** Everything under `data/processed/`.

**What happens:**

1. For each class in `CLASS_FOLDERS`, glob `images/*.png` (sorted).
2. Build one row per image:

   | column | example |
   | --- | --- |
   | `class` | `"COVID"` |
   | `image_path` | absolute path to the PNG |
   | `mask_path` | path to `{same basename}` in `masks/` |

**Output:** A pandas DataFrame with one row per image (~21k rows for the full dataset). This is the canonical list of samples for splitting and preprocessing. Image order is deterministic because paths are sorted.

### 3. Stratified split (`manifest.py`)

**Input:** The manifest DataFrame.

**What happens:**

1. First split: 70% train, 30% holdout (`val_size + test_size`, default 0.15 + 0.15).
2. Second split: holdout divided equally into val and test (50/50 of the 30%, so 15% / 15% overall).
3. Both splits use `sklearn.model_selection.train_test_split` with `stratify=manifest["class"]` and `random_state=42` by default.

**Output:** A `Splits` object with three DataFrames: `train`, `val`, `test`. Each row is still `(class, image_path, mask_path)`. Class proportions stay roughly equal across splits.

### 4. Per-image preprocessing (`transforms.py`)

For each row in a split, `preprocess_xray(image_path, mask_path, config)` runs:

```
read PNG as grayscale (uint8, H×W)
read mask as grayscale (uint8, H×W)
resize image → target_size (default 224×224), INTER_AREA
resize mask  → same size, INTER_NEAREST (keeps mask edges sharp)
[optional] augment image + mask together (augmentation.py)
[optional] CLAHE on image only
[optional] zero pixels outside lung mask (mask > 127)
normalize: float32, per-image min-max → [0, 1]
```

**CLAHE** (Contrast Limited Adaptive Histogram Equalization): improves local contrast in chest X-rays. Applied to the image only; the mask is not equalized.

**Lung mask:** Pixels where `mask <= 127` are set to 0 on the image. Background and non-lung regions are removed before normalization.

**Normalization:** `(pixel - min) / (max - min + 1e-8)`. Each image is scaled independently, so global intensity differences between scans are reduced.

**Output:** One 2D `float32` array of shape `(224, 224)` per image (or whatever `target_size` is).

### 5. Augmentation (`augmentation.py`)

Only runs when `PreprocessConfig.augment=True`. By default only the **train** split is augmented (`augment_splits=("train",)` in `run_preprocessing`).

Image and mask are transformed **together** so they stay aligned:

| operation | effect on image | effect on mask |
| --- | --- | --- |
| horizontal flip | `cv2.flip(..., 1)` | same |
| rotation + slight zoom | `warpAffine`, reflect borders | nearest neighbor, zero border |
| zoom | resize then crop or pad | same geometry |
| translation | shift with reflect | nearest neighbor |
| brightness/contrast | scale + offset on intensities | unchanged |

Each operation has a probability (defaults 0.5 for geometry, 0.4 for brightness). A seeded `random.Random` makes runs reproducible when `random_state` is fixed.

Augmentation runs **after** resize and **before** CLAHE and lung masking.

### 6. Stack into arrays (`dataset.py`)

**Input:** A split DataFrame (e.g. all train rows).

**What happens:**

1. Loop rows, call `preprocess_xray` for each.
2. On failure (missing file, corrupt PNG), record `(path, error)` in `failed` and continue.
3. Stack successful images into `X` with shape `(n_samples, height, width)`.
4. Collect class strings into `y` with shape `(n_samples,)`.

**Output:** `ArrayDataset(X, y, failed)`.

`build_split_arrays()` does this for train, val, and test. Val and test use the same `PreprocessConfig` except augmentation is turned off unless the split name is in `augment_splits`.

### 7. Validation (`validation.py`)

Before saving, the pipeline checks consistency:

1. **Count check:** number of PNGs in `data/processed/` == manifest rows == sum of samples across splits (minus any failed reads).
2. **Compare to previous save:** if `data/arrays/X_train.npy` already exists, load it and compare shape, labels, and pixel values to the fresh build. Reports max diff and how many samples changed.

This catches accidental data loss or config changes between runs.

### 8. Save (`dataset.py`, `pipeline.py`)

When `save=True`:

- `X_train.npy`, `y_train.npy`, and the same for `val` and `test`.
- `preprocessing.json`: timestamp, paths, all config dataclasses as dicts, per-split sample counts and class counts, failed counts, Python/numpy/opencv/sklearn versions.

Labels in `.npy` files are **strings** (`"COVID"`, `"Normal"`, …). Use `encode_labels()` to map to integers via `LABEL_TO_ID` from `../config.py`.

## Module map

| file | responsibility |
| --- | --- |
| `config.py` | `SplitConfig`, `PreprocessConfig` defaults for this step |
| `files.py` | redundancy CSV, copy raw → processed |
| `manifest.py` | manifest table, stratified train/val/test |
| `transforms.py` | single-image load, resize, CLAHE, mask, normalize |
| `augmentation.py` | paired random transforms for train |
| `dataset.py` | batch over manifest rows, stack arrays, save/load `.npy` |
| `validation.py` | count and diff checks against previous run |
| `pipeline.py` | `run_preprocessing`, metadata, `format_report` |
| `__main__.py` | CLI: `python -m covid_xray.preprocessing` or `covid-xray-preprocess` |

## Configuration

**SplitConfig** (`config.py`):

- `val_size=0.15`, `test_size=0.15` → 70 / 15 / 15 split
- `random_state=42` for reproducible assignment
- `stratify_column="class"`

**PreprocessConfig** (`config.py`):

- `target_size=(224, 224)`
- `apply_lung_mask=False`, `apply_clahe=False` by default
- `augment=False` by default
- `mask_threshold=127` for binarizing lung masks

**AugmentConfig** (`augmentation.py`): probabilities and ranges for each transform.

## How to run

From the project root (after `pip install -e .`):

```bash
# full run: copy, build, save
covid-xray-preprocess --redundant-csv notebooks/redundant_images.csv

# reuse processed folder, preview without writing
covid-xray-preprocess --skip-copy --dry-run

# enable CLAHE and augmentation on train only
covid-xray-preprocess --skip-copy --clahe --augment
```

From Python:

```python
from covid_xray.preprocessing import PreprocessConfig, format_report, run_preprocessing

result = run_preprocessing(
    redundant_csv="notebooks/redundant_images.csv",
    preprocess_config=PreprocessConfig(apply_clahe=True),
)
print(format_report(result))
```

## Relationship to training

This step materializes **eager** arrays: every image is preprocessed once and stored. That is simple and fast to load, but augmentation is fixed at build time and a full 224×224 float dataset is several GB.

A future training step may instead read the manifest and call `preprocess_xray` per batch (lazy loading) so augmentation can differ every epoch. The manifest and split logic in this package already support that; the `.npy` files are an optional cache, not the only interface.

## Tests

Tests live in `tests/preprocessing/` and mirror these modules. Run from the repo root:

```bash
pytest tests/preprocessing/
```

Synthetic 64×64 PNGs are generated in fixtures so tests do not need the real Kaggle download.
