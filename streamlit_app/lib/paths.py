from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "latex" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
APP_ASSETS_DIR = APP_DIR / "assets"
EXAMPLE_IMAGES_DIR = APP_ASSETS_DIR / "examples"
REPORT_PDF = REPORTS_DIR / "latex" / "main.pdf"


def example_image(class_dir: str, filename: str) -> Path:
    processed = DATA_DIR / "processed" / class_dir / "images" / filename
    if processed.exists():
        return processed
    bundled = EXAMPLE_IMAGES_DIR / filename
    if bundled.exists():
        return bundled
    return processed
