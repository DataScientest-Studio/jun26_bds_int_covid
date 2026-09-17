from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "latex" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_PDF = REPORTS_DIR / "latex" / "main.pdf"

