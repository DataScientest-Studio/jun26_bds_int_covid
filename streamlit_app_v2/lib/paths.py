from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_DIR = REPORTS_DIR / "baseline"
CNN_DIR = REPORTS_DIR / "cnn"
TRANSFER_DIR = REPORTS_DIR / "transfer_learning"
FIGURES_DIR = REPORTS_DIR / "figures"
LATEX_FIGURES_DIR = REPORTS_DIR / "latex" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

