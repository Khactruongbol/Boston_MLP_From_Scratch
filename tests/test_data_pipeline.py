from __future__ import annotations

from src.config import FEATURE_NAMES, TARGET_NAME
from src.data import clean_data, crawl_raw_data, reconstruct_boston_dataframe, split_features_target


def test_crawl_and_reconstruct_raw_boston_data():
    raw_text = crawl_raw_data()
    assert "Boston house-price data" in raw_text

    df = reconstruct_boston_dataframe(raw_text)
    assert df.shape == (506, 14)
    assert list(df.columns) == FEATURE_NAMES + [TARGET_NAME]


def test_clean_data_and_feature_split():
    raw_text = crawl_raw_data()
    df = reconstruct_boston_dataframe(raw_text)
    clean_df = clean_data(df)
    X, y = split_features_target(clean_df)

    assert clean_df.shape == (506, 14)
    assert X.shape == (506, 13)
    assert y.shape == (506,)
    assert not X.isna().any().any()
    assert not y.isna().any()

