from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd

from app import DEFAULT_VALUES, load_best_model, predict_price
from src.config import (
    METRICS_PATH,
    MODEL_METADATA_PATH,
    OPTIMIZED_METRICS_PATH,
    SCALER_PATH,
    TUNING_RESULTS_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "99_boston_housing_model_images_report.ipynb"


def test_project_does_not_use_removed_load_boston_api():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "src").glob("*.py"))
    assert "from sklearn.datasets" not in source
    assert "load_boston(" not in source


def test_tuning_does_not_use_test_set_for_hyperparameter_selection():
    modeling_source = (PROJECT_ROOT / "src" / "modeling.py").read_text(encoding="utf-8")
    assert "GridSearchCV" in modeling_source
    assert "validation_data=(X_test" not in modeling_source
    assert "search.fit(X_test" not in modeling_source


def test_required_project_files_exist():
    required_paths = [
        "src/data.py",
        "src/modeling.py",
        "src/train.py",
        "app.py",
        "notebooks/99_boston_housing_model_images_report.ipynb",
        "requirements.txt",
        "README.md",
    ]
    for relative_path in required_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_report_notebook_has_only_markdown_and_existing_images():
    notebook = json.loads(REPORT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert cells
    assert all(cell["cell_type"] == "markdown" for cell in cells)

    image_links: list[Path] = []
    for cell in cells:
        source = "".join(cell.get("source", []))
        for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source):
            image_links.append((REPORT_NOTEBOOK_PATH.parent / match).resolve())

    assert len(image_links) >= 8
    missing = [path for path in image_links if not path.exists()]
    assert not missing


def test_report_notebook_contains_required_vietnamese_sections():
    notebook = json.loads(REPORT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    report_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required_terms = [
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
    required_terms.append("Model optimization")
    for term in required_terms:
        assert term in report_text
    assert "\ufffd" not in report_text


def test_optimized_outputs_and_metadata_are_complete():
    metrics_df = pd.read_csv(METRICS_PATH)
    optimized_df = pd.read_csv(OPTIMIZED_METRICS_PATH)
    tuning_df = pd.read_csv(TUNING_RESULTS_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    assert metrics_df.shape == optimized_df.shape
    assert metrics_df.shape[0] >= 6
    assert metrics_df["model"].str.contains("Tuned").any()
    assert {"MSE", "RMSE", "MAE", "R2"}.issubset(metrics_df.columns)
    assert {"model", "candidate_type", "cv_rmse", "params"}.issubset(tuning_df.columns)
    assert len(tuning_df) >= 4
    assert metadata["selection_metric"] == "RMSE"
    assert "best_params" in metadata
    assert {"cv_rmse", "test_rmse", "test_r2", "features", "model_contains_scaler"}.issubset(metadata)
    assert metadata["test_rmse"] <= metadata["reference_rmse"] + 1e-12


def test_app_prediction_uses_saved_best_model():
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    model = load_best_model(metadata)
    scaler = None if metadata.get("model_contains_scaler") else joblib.load(SCALER_PATH)
    prediction = predict_price(DEFAULT_VALUES, model, scaler, bool(metadata.get("model_contains_scaler")))
    assert isinstance(prediction, float)
    assert prediction > 0
