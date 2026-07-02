from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import BEST_MODEL_KERAS_PATH, BEST_MODEL_PATH, FEATURE_NAMES, MODEL_METADATA_PATH, SCALER_PATH


DEFAULT_VALUES = {
    "CRIM": 0.1,
    "ZN": 0.0,
    "INDUS": 8.0,
    "CHAS": 0.0,
    "NOX": 0.5,
    "RM": 6.0,
    "AGE": 65.0,
    "DIS": 4.0,
    "RAD": 4.0,
    "TAX": 300.0,
    "PTRATIO": 18.0,
    "B": 390.0,
    "LSTAT": 12.0,
}


def load_metadata() -> dict:
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("Model metadata not found. Run `python -m src.train` before opening the app.")
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def load_best_model(metadata: dict):
    if metadata["best_model_type"] == "keras":
        import tensorflow as tf

        return tf.keras.models.load_model(BEST_MODEL_KERAS_PATH)
    return joblib.load(BEST_MODEL_PATH)


def predict_price(feature_values: dict[str, float], model, scaler=None, model_contains_scaler: bool = False) -> float:
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    model_input = input_df if model_contains_scaler else scaler.transform(input_df)
    prediction = model.predict(model_input)
    return float(np.asarray(prediction).reshape(-1)[0])


def main() -> None:
    st.set_page_config(page_title="Boston Housing Predictor", layout="centered")
    st.title("Boston Housing Price Prediction")

    try:
        metadata = load_metadata()
        model_contains_scaler = bool(metadata.get("model_contains_scaler", False))
        scaler = None if model_contains_scaler else joblib.load(SCALER_PATH)
        model = load_best_model(metadata)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"Best model: {metadata['best_model']} | selection metric: {metadata['selection_metric']}")

    feature_values: dict[str, float] = {}
    left, right = st.columns(2)
    for index, feature in enumerate(FEATURE_NAMES):
        container = left if index % 2 == 0 else right
        with container:
            feature_values[feature] = st.number_input(
                feature,
                value=float(DEFAULT_VALUES[feature]),
                step=0.1,
                format="%.4f",
            )

    if st.button("Predict MEDV"):
        predicted_value = predict_price(feature_values, model, scaler, model_contains_scaler)
        st.metric("Predicted MEDV", f"{predicted_value:.2f} thousand USD")

    st.subheader("Model Metrics")
    metrics_path = Path("reports") / "model_metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.info("Metrics file is not available yet. Run training first.")


if __name__ == "__main__":
    main()
