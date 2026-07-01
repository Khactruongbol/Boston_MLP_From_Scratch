from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    ANALYSIS_PATH,
    BEST_MODEL_KERAS_PATH,
    BEST_MODEL_PATH,
    FIGURE_DIR,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    PREDICTIONS_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    SCALER_PATH,
)


def validate_project() -> list[str]:
    checks: list[tuple[str, bool]] = []

    required_files = [
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        METRICS_PATH,
        PREDICTIONS_PATH,
        ANALYSIS_PATH,
        SCALER_PATH,
        MODEL_METADATA_PATH,
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "src" / "data.py",
        PROJECT_ROOT / "src" / "modeling.py",
        PROJECT_ROOT / "src" / "train.py",
    ]
    checks.extend((f"exists: {path.relative_to(PROJECT_ROOT)}", path.exists()) for path in required_files)

    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "src").glob("*.py"))
    forbidden_loader = "load_" + "boston("
    forbidden_import = "from sklearn." + "datasets"
    checks.append(("does not use sklearn load_boston", forbidden_loader not in source_text and forbidden_import not in source_text))

    clean_df = pd.read_csv(PROCESSED_DATA_PATH)
    checks.append(("clean data has 506 rows and 14 columns", clean_df.shape == (506, 14)))
    checks.append(("clean data has no missing values", not clean_df.isna().any().any()))

    metrics_df = pd.read_csv(METRICS_PATH)
    checks.append(("metrics include 4 trained models", metrics_df.shape[0] == 4))
    checks.append(("metrics include MSE RMSE MAE R2", set(["MSE", "RMSE", "MAE", "R2"]).issubset(metrics_df.columns)))

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    checks.append(("metadata records best model", bool(metadata.get("best_model"))))
    checks.append(("best model artifact exists", BEST_MODEL_PATH.exists() or BEST_MODEL_KERAS_PATH.exists()))
    checks.append(("at least 6 training figures exist", len(list(FIGURE_DIR.glob("*.png"))) >= 6))

    failed = [name for name, passed in checks if not passed]
    report_lines = ["Project validation report", ""]
    report_lines.extend(f"[{'PASS' if passed else 'FAIL'}] {name}" for name, passed in checks)
    report = "\n".join(report_lines)
    validation_path = PROJECT_ROOT / "reports" / "validation_report.txt"
    validation_path.write_text(report, encoding="utf-8")

    if failed:
        raise AssertionError("Validation failed: " + "; ".join(failed))
    return report_lines


def main() -> None:
    for line in validate_project():
        print(line)


if __name__ == "__main__":
    main()
