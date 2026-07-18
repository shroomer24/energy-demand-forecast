"""
src/train_xgboost.py
--------------------
Trains an XGBoost gradient boosting model on the engineered feature set.

XGBoost is the primary ML model — it handles nonlinear interactions between
temperature, time-of-day, day-of-week, and lag features naturally, making
it well suited for energy demand forecasting.

MLflow logs:
  - hyperparameters
  - feature importances
  - MAE, RMSE, MAPE on held-out test period
  - serialised model artifact
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import build_features, train_test_split_temporal
from src.tracker import ExperimentTracker

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data" / "demand.csv"
MODEL_PATH  = ROOT / "data" / "xgb_model.pkl"
FI_PATH     = ROOT / "data" / "xgb_feature_importance.json"
PRED_PATH   = ROOT / "data" / "xgb_predictions.csv"


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train():
    raw     = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    feat_df = build_features(raw)
    train_df, test_df = train_test_split_temporal(feat_df)

    exclude      = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    X_train = train_df[feature_cols].values
    y_train = train_df["demand_gw"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["demand_gw"].values

    print(f"[xgboost] Training model...")

    params = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model = XGBRegressor(**params)

    import time
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    preds    = model.predict(X_test)
    mae_val  = float(mean_absolute_error(y_test, preds))
    rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape_val = mape(y_test, preds)

    print(f"[xgboost] MAE={mae_val:.3f} GW  |  RMSE={rmse_val:.3f} GW  |  MAPE={mape_val:.2f}%")

    # Save model bundle
    bundle = {
        "model":        model,
        "feature_cols": feature_cols,
        "mape":         mape_val,
        "mae":          mae_val,
        "rmse":         rmse_val,
        "trained_at":   pd.Timestamp.now().isoformat(),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # Feature importance
    fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    with open(FI_PATH, "w") as f:
        json.dump(fi, f, indent=2)

    # Predictions
    pred_df = test_df[["timestamp", "demand_gw"]].copy()
    pred_df["xgb_pred"] = preds
    pred_df.to_csv(PRED_PATH, index=False)

    # Log to tracker
    tracker = ExperimentTracker("energy-demand-forecast", ROOT / "mlruns")
    tracker.log_run(
        run_name="xgboost_main",
        params={**params, "duration_s": round(duration, 2)},
        metrics={"mae": mae_val, "rmse": rmse_val, "mape": mape_val},
        tags={"model": "XGBoost", "stage": "production"},
    )

    print("\n[xgboost] Top 5 features by importance:")
    for k, v in list(fi.items())[:5]:
        print(f"  {k:<35} {v:.4f}")

    return {"model": "XGBoost", "mae": mae_val, "rmse": rmse_val, "mape": mape_val}


if __name__ == "__main__":
    train()
