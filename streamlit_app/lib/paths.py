from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARRAY_DIR = DATA_DIR / "arrays"
PREPROCESSING_JSON = ARRAY_DIR / "preprocessing.json"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_REPORTS_DIR = REPORTS_DIR / "baseline"
CNN_REPORTS_DIR = REPORTS_DIR / "cnn"
TRANSFER_REPORTS_DIR = REPORTS_DIR / "transfer_learning"
SUMMARY_FIGURES_DIR = REPORTS_DIR / "figures"

LATEX_DIR = REPORTS_DIR / "latex"
LATEX_FIGURES_DIR = LATEX_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
