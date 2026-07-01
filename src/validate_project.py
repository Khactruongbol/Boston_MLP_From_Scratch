from __future__ import annotations

import json
import re
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


REPORT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "99_boston_housing_model_images_report.ipynb"
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def notebook_image_paths(notebook_path: Path) -> list[Path]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    image_paths: list[Path] = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        for match in MARKDOWN_IMAGE_PATTERN.findall(source):
            if match.startswith(("http://", "https://")):
                continue
            image_paths.append((notebook_path.parent / match).resolve())
    return image_paths


def notebook_text(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


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
        REPORT_NOTEBOOK_PATH,
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

    notebook = json.loads(REPORT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook_cells = notebook.get("cells", [])
    report_text = notebook_text(REPORT_NOTEBOOK_PATH)
    image_paths = notebook_image_paths(REPORT_NOTEBOOK_PATH)
    checks.append(("report notebook has no code cells", all(cell.get("cell_type") != "code" for cell in notebook_cells)))
    checks.append(("report notebook contains model images", len(image_paths) >= 8))
    checks.append(("all report notebook image links exist", all(path.exists() for path in image_paths)))
    required_report_terms = [
        "Define problem",
        "Định nghĩa bài toán",
        "Hướng giải quyết",
        "Sử dụng những gì",
        "Data lấy từ đâu",
        "cách lấy data",
        "Cách lọc data sạch",
        "Số lượng feature: **13**",
        "Hình ảnh sau khi train",
        "Kết luận model",
    ]
    checks.append(("report notebook includes required explanation sections", all(term in report_text for term in required_report_terms)))
    checks.append(("report notebook text has valid Vietnamese UTF-8", "\ufffd" not in report_text))

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
