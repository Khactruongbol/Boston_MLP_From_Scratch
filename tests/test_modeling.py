from __future__ import annotations

import numpy as np

from sklearn.pipeline import Pipeline

from src.modeling import evaluate_predictions, make_sklearn_pipeline


def test_evaluate_predictions_returns_required_metrics():
    metrics = evaluate_predictions("demo", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.5]))
    assert set(metrics) == {"model", "MSE", "RMSE", "MAE", "R2"}
    assert metrics["model"] == "demo"
    assert metrics["MSE"] >= 0
    assert metrics["RMSE"] >= 0


def test_sklearn_pipeline_contains_scaler_before_model():
    from sklearn.ensemble import RandomForestRegressor

    pipeline = make_sklearn_pipeline(RandomForestRegressor(random_state=42))
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]
