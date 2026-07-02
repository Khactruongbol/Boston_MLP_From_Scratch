from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
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
    OPTIMIZED_METRICS_PATH,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    SCALER_PATH,
    TUNING_RESULTS_PATH,
)


REFERENCE_RMSE = 2.808917571159508


@dataclass
class TrainedModel:
    name: str
    model: Any
    predictions: np.ndarray
    model_type: str = "sklearn_pipeline"
    history: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    cv_rmse: float | None = None
    model_contains_scaler: bool = True


def prepare_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for the MLP model. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    tf.get_logger().setLevel("ERROR")
    return tf, Sequential, Dense, Input, Adam, EarlyStopping


def make_sklearn_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_mlp_model(input_dim: int, hidden_layers: tuple[int, ...] = (32, 16), learning_rate: float = 0.001):
    _, Sequential, Dense, Input, Adam, _ = import_tensorflow()
    layers = [Input(shape=(input_dim,))]
    layers.extend(Dense(units, activation="relu") for units in hidden_layers)
    layers.append(Dense(1))
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error", metrics=["mae"])
    return model


def evaluate_predictions(model_name: str, y_true, y_pred) -> dict[str, float | str]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_sklearn_baselines(X_train, X_test, y_train) -> list[TrainedModel]:
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    trained: list[TrainedModel] = []
    for name, model in baseline_models.items():
        pipeline = make_sklearn_pipeline(model)
        pipeline.fit(X_train, y_train)
        trained.append(
            TrainedModel(
                name=name,
                model=pipeline,
                predictions=pipeline.predict(X_test),
                params=serializable_model_params(model),
            )
        )
    return trained


def fit_tuned_sklearn_models(X_train, X_test, y_train) -> tuple[list[TrainedModel], pd.DataFrame]:
    tuning_specs = [
        (
            "Decision Tree Tuned",
            make_sklearn_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "Random Forest Tuned",
            make_sklearn_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": [1.0],
            },
        ),
    ]

    tuned_models: list[TrainedModel] = []
    tuning_rows: list[dict[str, Any]] = []
    for model_name, pipeline, param_grid in tuning_specs:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        cv_rmse = float(-search.best_score_)
        best_params = {key: normalize_param_value(value) for key, value in search.best_params_.items()}
        tuning_rows.append(
            {
                "model": model_name,
                "candidate_type": "GridSearchCV",
                "cv_rmse": cv_rmse,
                "params": json.dumps(best_params, sort_keys=True),
            }
        )
        tuned_models.append(
            TrainedModel(
                name=model_name,
                model=best_estimator,
                predictions=best_estimator.predict(X_test),
                params=best_params,
                cv_rmse=cv_rmse,
            )
        )

    return tuned_models, pd.DataFrame(tuning_rows)


def normalize_param_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_param_value(item) for item in value]
    return value


def serializable_model_params(model) -> dict[str, Any]:
    return {key: normalize_param_value(value) for key, value in model.get_params(deep=False).items()}


def json_safe_metric_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def fit_mlp_models(X_train, X_test, y_train) -> tuple[list[TrainedModel], pd.DataFrame, StandardScaler]:
    tf, _, _, _, _, EarlyStopping = import_tensorflow()
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    configs = [
        {
            "name": "MLP Regression",
            "hidden_layers": (32, 16),
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping": False,
        },
        {
            "name": "MLP Tuned",
            "hidden_layers": (64, 32),
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 200,
            "early_stopping": True,
        },
    ]

    trained: list[TrainedModel] = []
    tuning_rows: list[dict[str, Any]] = []
    for config in configs:
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        model = build_mlp_model(
            input_dim=X_train_scaled.shape[1],
            hidden_layers=config["hidden_layers"],
            learning_rate=config["learning_rate"],
        )
        callbacks = []
        if config["early_stopping"]:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True))
        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )
        val_rmse = float(np.sqrt(min(history.history["val_loss"])))
        params = {
            "hidden_layers": list(config["hidden_layers"]),
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "early_stopping": config["early_stopping"],
        }
        tuning_rows.append(
            {
                "model": config["name"],
                "candidate_type": "train_validation_split",
                "cv_rmse": val_rmse,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        predictions = model.predict(X_test_scaled, verbose=0).reshape(-1)
        trained.append(
            TrainedModel(
                name=config["name"],
                model=model,
                predictions=predictions,
                model_type="keras",
                history=history,
                params=params,
                cv_rmse=val_rmse,
                model_contains_scaler=False,
            )
        )

    return trained, pd.DataFrame(tuning_rows), scaler


def train_models(X_train, X_test, y_train, y_test=None) -> tuple[list[TrainedModel], pd.DataFrame]:
    del y_test
    sklearn_models = fit_sklearn_baselines(X_train, X_test, y_train)
    tuned_sklearn_models, sklearn_tuning = fit_tuned_sklearn_models(X_train, X_test, y_train)
    mlp_models, mlp_tuning, _ = fit_mlp_models(X_train, X_test, y_train)
    tuning_df = pd.concat([sklearn_tuning, mlp_tuning], ignore_index=True)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False)
    return [*mlp_models, *sklearn_models, *tuned_sklearn_models], tuning_df


def evaluate_models(trained_models: list[TrainedModel], y_test) -> pd.DataFrame:
    results = [evaluate_predictions(item.name, y_test, item.predictions) for item in trained_models]
    metadata_rows = {
        item.name: {
            "CV_RMSE": item.cv_rmse,
            "model_type": item.model_type,
        }
        for item in trained_models
    }
    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    metrics_df["CV_RMSE"] = metrics_df["model"].map(lambda name: metadata_rows[name]["CV_RMSE"])
    metrics_df["model_type"] = metrics_df["model"].map(lambda name: metadata_rows[name]["model_type"])
    metrics_df.to_csv(METRICS_PATH, index=False)
    metrics_df.to_csv(OPTIMIZED_METRICS_PATH, index=False)
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
    plt.figure(figsize=(11, 5))
    plt.bar(metrics_df["model"], metrics_df["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Model RMSE Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_training_curve(history, model_name: str = "MLP") -> Path | None:
    if history is None:
        return None
    filename = "mlp_training_curve.png" if model_name == "MLP Regression" else f"{safe_name(model_name)}_training_curve.png"
    path = FIGURE_DIR / filename
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train MSE")
    plt.plot(history.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{model_name} Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def get_feature_importances(best_model: TrainedModel) -> np.ndarray | None:
    model = best_model.model
    if isinstance(model, Pipeline):
        model = model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    return np.asarray(model.feature_importances_)


def save_feature_importance_plot(best_model: TrainedModel) -> Path | None:
    importances = get_feature_importances(best_model)
    if importances is None:
        return None
    path = FIGURE_DIR / "best_model_feature_importance.png"
    order = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(FEATURE_NAMES)[order], importances[order])
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best_model.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_tuning_summary_plot(tuning_df: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "tuning_summary.png"
    display_df = tuning_df.copy()
    display_df["cv_rmse"] = display_df["cv_rmse"].map(lambda value: f"{value:.4f}")
    display_df["params"] = display_df["params"].map(lambda value: value[:90] + "..." if len(value) > 90 else value)

    fig_height = max(3.2, 0.6 * (len(display_df) + 1))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df[["model", "candidate_type", "cv_rmse", "params"]].values,
        colLabels=["model", "candidate_type", "cv_rmse", "params"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("Tuning Summary", fontsize=16, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metrics_table_plot(metrics_df: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "model_metrics_table.png"
    display_df = metrics_df[["model", "MSE", "RMSE", "MAE", "R2", "CV_RMSE"]].copy()
    for column in ["MSE", "RMSE", "MAE", "R2", "CV_RMSE"]:
        display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")

    fig, ax = plt.subplots(figsize=(12, 0.55 * (len(display_df) + 2)))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#dcfce7")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("Model Metrics on Test Set", fontsize=16, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_all_figures(
    trained_models: list[TrainedModel],
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    y_test,
) -> list[Path]:
    figure_paths: list[Path] = []
    for item in trained_models:
        figure_paths.append(save_actual_vs_predicted_plot(item.name, y_test, item.predictions))
        if item.history is not None:
            curve_path = save_training_curve(item.history, item.name)
            if curve_path is not None:
                figure_paths.append(curve_path)
    figure_paths.append(save_model_comparison_plot(metrics_df))
    figure_paths.append(save_metrics_table_plot(metrics_df))
    figure_paths.append(save_tuning_summary_plot(tuning_df))
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


def save_best_model(
    best_model: TrainedModel,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    figure_paths: list[Path],
) -> dict[str, Any]:
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
    best_tuning_rows = tuning_df.loc[tuning_df["model"] == best_model.name]
    best_tuning = best_tuning_rows.iloc[0].to_dict() if not best_tuning_rows.empty else {}
    baseline_rmse = float(metrics_df.loc[metrics_df["model"] == "Random Forest Regression", "RMSE"].iloc[0])
    best_rmse = float(best_row["RMSE"])
    optimization_outcome = (
        "Optimized candidate improved over the Random Forest baseline."
        if best_rmse < baseline_rmse
        else "No tuned candidate improved over the Random Forest baseline; retained the strongest held-out test model."
    )

    metadata = {
        "best_model": best_model.name,
        "best_model_type": best_model.model_type,
        "best_model_path": str(model_path.relative_to(MODEL_DIR.parent)),
        "scaler_path": str(SCALER_PATH.relative_to(MODEL_DIR.parent)),
        "model_contains_scaler": best_model.model_contains_scaler,
        "selection_metric": "RMSE",
        "selection_rule": "Lowest RMSE on the held-out test set after train-only tuning",
        "reference_rmse": REFERENCE_RMSE,
        "optimization_outcome": optimization_outcome,
        "best_params": best_model.params,
        "cv_rmse": None if pd.isna(best_row.get("CV_RMSE")) else float(best_row["CV_RMSE"]),
        "test_rmse": best_rmse,
        "test_r2": float(best_row["R2"]),
        "metrics": {
            key: json_safe_metric_value(value)
            for key, value in best_row.items()
        },
        "tuning": best_tuning,
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
            "Optimization: sklearn models use train-only GridSearchCV pipelines, and MLP candidates use a train-validation split with early stopping where configured.",
            "Model selection: the best model is selected by the lowest RMSE on the held-out test set after tuning is complete.",
            "",
            f"Best model by RMSE: {best_row['model']} (RMSE={best_row['RMSE']:.4f}, R2={best_row['R2']:.4f}).",
            f"Best model by R2: {best_r2_row['model']} (R2={best_r2_row['R2']:.4f}, RMSE={best_r2_row['RMSE']:.4f}).",
            f"MLP Regression: MSE={mlp_row['MSE']:.4f}, RMSE={mlp_row['RMSE']:.4f}, "
            f"MAE={mlp_row['MAE']:.4f}, R2={mlp_row['R2']:.4f}.",
            metadata["optimization_outcome"],
            "",
            f"Saved best model: {metadata['best_model_path']}",
            f"Saved scaler: {metadata['scaler_path']}",
        ]
    )
    ANALYSIS_PATH.write_text(analysis, encoding="utf-8")
    return analysis
