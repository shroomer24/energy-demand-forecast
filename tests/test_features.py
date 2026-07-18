"""
tests/test_features.py
----------------------
Unit tests for src/features.py — feature engineering correctness.
Run: pytest tests/test_features.py
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import build_features

REQUIRED_COLS = [
    "hour", "dayofweek", "month", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
    "temp_sq", "temp_cooling", "temp_heating",
    "demand_lag_1h", "demand_lag_2h", "demand_lag_3h", "demand_lag_6h",
    "demand_lag_12h", "demand_lag_24h", "demand_lag_48h",
    "demand_lag_72h", "demand_lag_168h",
    "demand_roll_mean_24h", "demand_roll_std_24h", "demand_roll_mean_168h",
    "data_center_load_gw",
]


def _make_raw(n=300):
    return pd.DataFrame({
        "timestamp":     pd.date_range("2023-01-01", periods=n, freq="h"),
        "demand_gw":     np.random.uniform(60, 150, n),
        "temperature_f": np.random.uniform(20, 100, n),
    })


def test_all_required_columns_present():
    feat = build_features(_make_raw())
    for col in REQUIRED_COLS:
        assert col in feat.columns, f"Missing feature column: {col}"


def test_no_nans_after_dropna():
    feat = build_features(_make_raw())
    null_counts = feat[REQUIRED_COLS].isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found:\n{null_counts[null_counts > 0]}"


def test_cyclic_features_bounded():
    feat = build_features(_make_raw())
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"]:
        assert feat[col].between(-1.0001, 1.0001).all(), \
            f"{col} has values outside [-1, 1]"


def test_is_weekend_binary():
    feat = build_features(_make_raw())
    assert set(feat["is_weekend"].unique()).issubset({0, 1})


def test_lag_features_shift_correctly():
    """demand_lag_1h at row i should equal demand_gw at row i-1."""
    raw = _make_raw(300)
    feat = build_features(raw).reset_index(drop=True)
    # Compare lag_1h against the shifted demand column
    aligned = feat[["demand_gw", "demand_lag_1h"]].copy()
    # demand_lag_1h[i] == demand_gw[i-1] for rows where both are valid
    diffs = (aligned["demand_lag_1h"] - aligned["demand_gw"].shift(1)).dropna()
    assert (diffs.abs() < 1e-6).all(), "demand_lag_1h does not match shifted demand_gw"


def test_temp_cooling_non_negative():
    feat = build_features(_make_raw())
    assert (feat["temp_cooling"] >= 0).all()


def test_temp_heating_non_negative():
    feat = build_features(_make_raw())
    assert (feat["temp_heating"] >= 0).all()


def test_row_count_reduced_by_dropna():
    """build_features drops NaN rows from lag creation — output must be smaller than input."""
    raw = _make_raw(300)
    feat = build_features(raw)
    assert len(feat) < len(raw), "Expected rows to be dropped due to lag NaNs"


def test_data_center_load_increases_over_time():
    """data_center_load_gw should grow linearly from 2022 onward."""
    raw = _make_raw(8760)  # 1 year
    feat = build_features(raw)
    first_half = feat["data_center_load_gw"].iloc[:len(feat)//2].mean()
    second_half = feat["data_center_load_gw"].iloc[len(feat)//2:].mean()
    assert second_half > first_half, "data_center_load_gw should increase over time"
