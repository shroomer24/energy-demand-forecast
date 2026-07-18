"""
src/train_intervals.py
-----------------------
Trains quantile regression models to produce 80% prediction intervals,
then applies conformal prediction calibration to guarantee coverage.

Why conformal calibration?
  Quantile regression targets 10th/90th percentiles in expectation but can
  under-cover (62% empirical vs 80% target) due to distributional shift.
  Conformal prediction fixes this distribution-free: it computes non-conformity
  scores on a held-out calibration set and inflates intervals until empirical
  coverage matches the target exactly.

Output: data/interval_models.pkl
  Includes: lower/median/upper quantile models + conformal_offset (GW)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost import XGBRegressor
from src.features import build_features, train_test_split_temporal

ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "demand.csv"
MODEL_PATH = ROOT / "data" / "interval_models.pkl"


def train_quantile_models():
    print("[intervals] Loading features ...")
    raw     = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    feat_df = build_features(raw)

    exclude      = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    # ── Temporal split: train | calibration | test ────────────────────────────
    # Use last 720 hours of training data as conformal calibration set
    train_full, test_df = train_test_split_temporal(feat_df)
    calib_df  = train_full.iloc[-720:]    # last 720h = 30 days for calibration
    train_df  = train_full.iloc[:-720]    # remaining for quantile model training

    X_train = train_df[feature_cols].values
    y_train = train_df["demand_gw"].values
    X_calib = calib_df[feature_cols].values
    y_calib = calib_df["demand_gw"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["demand_gw"].values

    models    = {}
    quantiles = {"lower": 0.10, "median": 0.50, "upper": 0.90}
    calib_lower = calib_upper = None

    for name, q in quantiles.items():
        print(f"[intervals] Training {name} model  (q={q}) ...")
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"[intervals]   range: {preds.min():.1f} – {preds.max():.1f} GW")

        if name == "lower":
            lower_preds  = preds
            calib_lower  = model.predict(X_calib)
        elif name == "upper":
            upper_preds  = preds
            calib_upper  = model.predict(X_calib)

        models[name] = {"model": model, "feature_cols": feature_cols, "quantile": q}

    # ── Raw coverage before calibration ───────────────────────────────────────
    raw_coverage = float(
        np.mean((y_test >= lower_preds) & (y_test <= upper_preds)) * 100
    )
    print(f"\n[intervals] Raw coverage (before calibration): {raw_coverage:.1f}%  (target ≈ 80%)")

    # ── Conformal prediction calibration ──────────────────────────────────────
    # Non-conformity score: how far outside the interval each calibration point is
    # s_i = max(y_lower_i - y_i, y_i - y_upper_i)
    # Negative = inside interval, positive = outside
    nonconformity = np.maximum(calib_lower - y_calib, y_calib - calib_upper)

    # Find the 80th percentile of non-conformity scores
    # This is the smallest offset q_hat such that 80% of calibration points
    # fall within [lower - q_hat, upper + q_hat]
    target_coverage = 0.80
    n_calib = len(nonconformity)
    q_level = np.ceil((1 + n_calib) * target_coverage) / n_calib
    q_level = min(q_level, 1.0)
    conformal_offset = float(np.quantile(nonconformity, q_level))

    # ── Calibrated coverage on test set ───────────────────────────────────────
    cal_lower = lower_preds - conformal_offset
    cal_upper = upper_preds + conformal_offset
    cal_coverage = float(
        np.mean((y_test >= cal_lower) & (y_test <= cal_upper)) * 100
    )
    print(f"[intervals] Conformal offset   : {conformal_offset:+.3f} GW")
    print(f"[intervals] Calibrated coverage: {cal_coverage:.1f}%  (target ≈ 80%)")

    # Store conformal offset so the API can apply it at inference time
    models["conformal_offset"] = conformal_offset

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"[intervals] Saved interval_models.pkl")

    return models


if __name__ == "__main__":
    train_quantile_models()
