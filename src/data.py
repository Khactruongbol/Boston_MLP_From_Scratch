from __future__ import annotations

import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BOSTON_DATA_URL,
    FEATURE_NAMES,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    TARGET_NAME,
)


def crawl_raw_data(url: str = BOSTON_DATA_URL, output_path=RAW_DATA_PATH, retries: int = 3) -> str:
    """Download the required Boston Housing raw text file and save it locally."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=90) as response:
                content = response.read().decode("utf-8")
            output_path.write_text(content, encoding="utf-8")
            return content
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    if output_path.exists():
        return output_path.read_text(encoding="utf-8")

    raise RuntimeError(
        "Could not crawl Boston Housing raw data from StatLib after retrying. "
        "Check the network connection and keep the required data source."
    ) from last_error


def reconstruct_boston_dataframe(raw_text: str) -> pd.DataFrame:
    """Parse StatLib raw text into a dataframe with 13 features and MEDV target."""
    rows: list[list[float]] = []
    for line in raw_text.splitlines()[22:]:
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.split()])

    if len(rows) != 1012:
        raise ValueError(f"Unexpected raw StatLib row count: {len(rows)}; expected 1012.")
    if any(len(rows[index]) != 11 for index in range(0, len(rows), 2)):
        raise ValueError("Unexpected StatLib format: every first sample row must contain 11 values.")
    if any(len(rows[index]) != 3 for index in range(1, len(rows), 2)):
        raise ValueError("Unexpected StatLib format: every second sample row must contain 3 values.")

    first_rows = np.array(rows[::2], dtype=float)
    second_rows = np.array(rows[1::2], dtype=float)
    data = np.hstack([first_rows, second_rows[:, :2]])
    target = second_rows[:, 2]

    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    df[TARGET_NAME] = target
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the reconstructed Boston Housing dataframe."""
    expected_columns = FEATURE_NAMES + [TARGET_NAME]
    if list(df.columns) != expected_columns:
        raise ValueError("Boston dataframe schema does not match the expected feature and target columns.")

    clean_df = df.copy()
    for column in expected_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.drop_duplicates().dropna().reset_index(drop=True)

    if clean_df.shape != (506, 14):
        raise ValueError(f"Unexpected clean data shape: {clean_df.shape}; expected (506, 14).")
    if clean_df[FEATURE_NAMES].isna().any().any() or clean_df[TARGET_NAME].isna().any():
        raise ValueError("Cleaned Boston data still contains missing values.")
    if (clean_df[TARGET_NAME] <= 0).any():
        raise ValueError("Target MEDV must contain positive house price values.")

    return clean_df


def save_clean_data(clean_df: pd.DataFrame, output_path=PROCESSED_DATA_PATH) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)


def split_features_target(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = clean_df[FEATURE_NAMES].copy()
    y = clean_df[TARGET_NAME].copy()
    return X, y


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def build_dataset_from_raw() -> pd.DataFrame:
    raw_text = crawl_raw_data()
    reconstructed_df = reconstruct_boston_dataframe(raw_text)
    clean_df = clean_data(reconstructed_df)
    save_clean_data(clean_df)
    return clean_df
