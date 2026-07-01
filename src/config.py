from __future__ import annotations

from pathlib import Path


RANDOM_STATE = 42
BOSTON_DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"

FEATURE_NAMES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
TARGET_NAME = "MEDV"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "boston_raw.txt"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "boston_clean.csv"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
ANALYSIS_PATH = REPORT_DIR / "analysis.txt"

SCALER_PATH = MODEL_DIR / "scaler.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
BEST_MODEL_KERAS_PATH = MODEL_DIR / "best_model.keras"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

