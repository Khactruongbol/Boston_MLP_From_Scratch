from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ANALYSIS_PATH,
    BEST_MODEL_KERAS_PATH,
    BEST_MODEL_PATH,
    FIGURE_DIR,
    FEATURE_NAMES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    SCALER_PATH,
)


@dataclass
class TrainedModel:
    name: str
    model: Any
    predictions: np.ndarray
    history: Any | None = None
    model_type: str = "sklearn"


def prepare_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for the MLP model. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    tf.get_logger().setLevel("ERROR")
    return tf, Sequential, Dense, Input


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    return scaler, X_train_scaled, X_test_scaled


def build_mlp_model(input_dim: int):
    _, Sequential, Dense, Input = import_tensorflow()
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def train_models(X_train_scaled, X_test_scaled, y_train, y_test) -> list[TrainedModel]:
    tf, _, _, _ = import_tensorflow()
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    trained: list[TrainedModel] = []

    mlp_model = build_mlp_model(input_dim=X_train_scaled.shape[1])
    history = mlp_model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=0,
    )
    mlp_pred = mlp_model.predict(X_test_scaled, verbose=0).reshape(-1)
    trained.append(
        TrainedModel(
            name="MLP Regression",
            model=mlp_model,
            predictions=mlp_pred,
            history=history,
            model_type="keras",
        )
    )

    sklearn_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    for name, model in sklearn_models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        trained.append(TrainedModel(name=name, model=model, predictions=predictions))

    return trained


def evaluate_predictions(model_name: str, y_true, y_pred) -> dict[str, float | str]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(trained_models: list[TrainedModel], y_test) -> pd.DataFrame:
    results = [evaluate_predictions(item.name, y_test, item.predictions) for item in trained_models]
    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    return metrics_df


def select_best_model(trained_models: list[TrainedModel], metrics_df: pd.DataFrame) -> TrainedModel:
    best_name = str(metrics_df.iloc[0]["model"])
    for item in trained_models:
        if item.name == best_name:
            return item
    raise ValueError(f"Best model {best_name} was not found in trained model objects.")


def safe_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("/", "_")


def save_actual_vs_predicted_plot(model_name: str, y_test, predictions: np.ndarray) -> Path:
    path = FIGURE_DIR / f"actual_vs_predicted_{safe_name(model_name)}.png"
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.75)
    min_value = min(float(np.min(y_test)), float(np.min(predictions)))
    max_value = max(float(np.max(y_test)), float(np.max(predictions)))
    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Ideal prediction")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "model_rmse_comparison.png"
    plt.figure(figsize=(9, 5))
    plt.bar(metrics_df["model"], metrics_df["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Model RMSE Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_training_curve(history) -> Path | None:
    if history is None:
        return None
    path = FIGURE_DIR / "mlp_training_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train MSE")
    plt.plot(history.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MLP Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_feature_importance_plot(best_model: TrainedModel) -> Path | None:
    if not hasattr(best_model.model, "feature_importances_"):
        return None
    path = FIGURE_DIR / "best_model_feature_importance.png"
    importances = np.asarray(best_model.model.feature_importances_)
    order = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(FEATURE_NAMES)[order], importances[order])
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best_model.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_all_figures(trained_models: list[TrainedModel], metrics_df: pd.DataFrame, y_test) -> list[Path]:
    figure_paths: list[Path] = []
    for item in trained_models:
        figure_paths.append(save_actual_vs_predicted_plot(item.name, y_test, item.predictions))
        if item.history is not None:
            curve_path = save_training_curve(item.history)
            if curve_path is not None:
                figure_paths.append(curve_path)
    figure_paths.append(save_model_comparison_plot(metrics_df))
    best_model = select_best_model(trained_models, metrics_df)
    importance_path = save_feature_importance_plot(best_model)
    if importance_path is not None:
        figure_paths.append(importance_path)
    return figure_paths


def save_predictions(trained_models: list[TrainedModel], y_test) -> pd.DataFrame:
    predictions_df = pd.DataFrame({"Actual": np.asarray(y_test)})
    for item in trained_models:
        predictions_df[item.name] = item.predictions
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    return predictions_df


def save_best_model(best_model: TrainedModel, metrics_df: pd.DataFrame, figure_paths: list[Path]) -> dict[str, Any]:
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()
    if BEST_MODEL_KERAS_PATH.exists():
        BEST_MODEL_KERAS_PATH.unlink()

    if best_model.model_type == "keras":
        best_model.model.save(BEST_MODEL_KERAS_PATH)
        model_path = BEST_MODEL_KERAS_PATH
    else:
        joblib.dump(best_model.model, BEST_MODEL_PATH)
        model_path = BEST_MODEL_PATH

    best_row = metrics_df.iloc[0].to_dict()
    metadata = {
        "best_model": best_model.name,
        "best_model_type": best_model.model_type,
        "best_model_path": str(model_path.relative_to(MODEL_DIR.parent)),
        "scaler_path": str(SCALER_PATH.relative_to(MODEL_DIR.parent)),
        "selection_metric": "RMSE",
        "selection_rule": "Lowest RMSE on the held-out test set",
        "metrics": {key: float(value) if key != "model" else value for key, value in best_row.items()},
        "figures": [str(path.relative_to(MODEL_DIR.parent)) for path in figure_paths],
        "features": FEATURE_NAMES,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_analysis(metrics_df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    best_row = metrics_df.iloc[0]
    best_r2_row = metrics_df.sort_values("R2", ascending=False).iloc[0]
    mlp_row = metrics_df.loc[metrics_df["model"] == "MLP Regression"].iloc[0]

    analysis = "\n".join(
        [
            "Boston Housing Regression Training Analysis",
            "",
            "Data source: raw Boston Housing data crawled from StatLib/CMU.",
            "Cleaning: numeric conversion, duplicate removal, missing-value removal, schema validation.",
            "Feature split: 13 Boston Housing features are used to predict MEDV.",
            "Model selection: the best model is selected by the lowest RMSE on the same held-out test set.",
            "",
            f"Best model by RMSE: {best_row['model']} (RMSE={best_row['RMSE']:.4f}, R2={best_row['R2']:.4f}).",
            f"Best model by R2: {best_r2_row['model']} (R2={best_r2_row['R2']:.4f}, RMSE={best_r2_row['RMSE']:.4f}).",
            f"MLP Regression: MSE={mlp_row['MSE']:.4f}, RMSE={mlp_row['RMSE']:.4f}, "
            f"MAE={mlp_row['MAE']:.4f}, R2={mlp_row['R2']:.4f}.",
            "",
            f"Saved best model: {metadata['best_model_path']}",
            f"Saved scaler: {metadata['scaler_path']}",
        ]
    )
    ANALYSIS_PATH.write_text(analysis, encoding="utf-8")
    return analysis
