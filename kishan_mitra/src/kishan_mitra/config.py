
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "Datasets"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_PATH = (
    DATASETS_DIR / "Custom_Crops_yield_Historical_Dataset.csv"
)

TARGET_COLUMN = "Yield_kg_per_ha"
RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_ARTIFACT_PATH = MODELS_DIR / "crop_yield_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
