"""
src/features.py
---------------
Feature engineering for electricity demand forecasting.

Generates:
  1. Calendar features  — hour, day-of-week, month, is_weekend, is_holiday
  2. Cyclic encodings   — sin/cos for hour, dow, month, dayofyear
  3. Lag features       — demand at t-1h, t-2h, t-3h, t-6h, t-12h, t-24h, t-48h, t-72h, t-168h
  4. Rolling statistics — 24h and 168h rolling mean & std
  5. Temperature        — raw, squared, cooling degree hours, heating degree hours
  6. Data center proxy  — linear growth trend in load
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT     = Path(__file__).parent.parent
IN_PATH  = ROOT / "data" / "demand.csv"
OUT_PATH = ROOT / "data" / "features.csv"

US_HOLIDAYS = {
    "2022-01-17","2022-02-21","2022-05-30","2022-07-04",
    "2022-09-05","2022-11-24","2022-12-26",
    "2023-01-02","2023-01-16","2023-02-20","2023-05-29","2023-07-04",
    "2023-09-04","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27",
    "2024-07-04","2024-09-02","2024-11-28","2024-12-25",
}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    feat = raw.copy()
    if "timestamp" in feat.columns:
        feat = feat.sort_values("timestamp").reset_index(drop=True)
    else:
        feat = feat.sort_index()

    ts = feat["timestamp"] if "timestamp" in feat.columns else feat.index

    feat["hour"]        = ts.dt.hour
    feat["dayofweek"]   = ts.dt.dayofweek
    feat["month"]       = ts.dt.month
    feat["dayofyear"]   = ts.dt.dayofyear
    feat["weekofyear"]  = ts.dt.isocalendar().week.astype(int).values
    feat["quarter"]     = ts.dt.quarter
    feat["is_weekend"]  = (feat["dayofweek"] >= 5).astype(int)
    feat["is_holiday"]  = ts.dt.strftime("%Y-%m-%d").isin(US_HOLIDAYS).astype(int)

    # Cyclic encodings
    feat["hour_sin"]       = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"]       = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow_sin"]        = np.sin(2 * np.pi * feat["dayofweek"] / 7)
    feat["dow_cos"]        = np.cos(2 * np.pi * feat["dayofweek"] / 7)
    feat["month_sin"]      = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"]      = np.cos(2 * np.pi * feat["month"] / 12)
    feat["dayofyear_sin"]  = np.sin(2 * np.pi * feat["dayofyear"] / 365)
    feat["dayofyear_cos"]  = np.cos(2 * np.pi * feat["dayofyear"] / 365)

    # Temperature features
    temp_col = "temperature_f" if "temperature_f" in feat.columns else "temp_f"
    if temp_col in feat.columns:
        t = feat[temp_col]
        feat["temp_sq"]        = t ** 2
        feat["temp_cooling"]   = np.maximum(0, t - 65)
        feat["temp_heating"]   = np.maximum(0, 45 - t)
    else:
        feat["temp_sq"] = feat["temp_cooling"] = feat["temp_heating"] = 0.0

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        feat[f"demand_lag_{lag}h"] = feat["demand_gw"].shift(lag)

    # Rolling statistics
    feat["demand_roll_mean_24h"]  = feat["demand_gw"].shift(1).rolling(24).mean()
    feat["demand_roll_std_24h"]   = feat["demand_gw"].shift(1).rolling(24).std()
    feat["demand_roll_mean_168h"] = feat["demand_gw"].shift(1).rolling(168).mean()

    # Data center load growth proxy (linear trend 2022-2024)
    if "timestamp" in feat.columns:
        t0 = pd.Timestamp("2022-01-01")
        days_elapsed = (feat["timestamp"] - t0).dt.days
    else:
        days_elapsed = pd.Series(range(len(feat)), index=feat.index)
    feat["data_center_load_gw"] = 2.0 + (days_elapsed / 365) * 0.75

    feat = feat.dropna()
    return feat


def train_test_split_temporal(feat_df: pd.DataFrame, test_start: str = "2024-10-01"):
    train = feat_df[feat_df["timestamp"] < test_start].copy()
    test  = feat_df[feat_df["timestamp"] >= test_start].copy()
    return train, test


if __name__ == "__main__":
    raw     = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    feat_df = build_features(raw)
    train, test = train_test_split_temporal(feat_df)

    print(f"[features] Train: {train['timestamp'].min().date()} → {train['timestamp'].max().date()} ({len(train):,} rows)")
    print(f"[features] Test:  {test['timestamp'].min().date()}  → {test['timestamp'].max().date()}  ({len(test):,} rows)")

    exclude = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    fcols   = [c for c in feat_df.columns if c not in exclude]
    print(f"[features] Feature count: {len(fcols)}")

    feat_df.to_csv(OUT_PATH, index=False)
    print(f"[features] Saved features.csv — {feat_df.shape}")
