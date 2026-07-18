"""
src/retrain.py
--------------
Automated retraining pipeline for production deployment.

This script is designed to run on a schedule (e.g., weekly cron job or
Airflow DAG) to keep the model current as new grid data arrives.

Flow:
  1. Fetch latest EIA data since the last successful run
  2. Merge with existing historical data
  3. Retrain XGBoost on the expanded dataset
  4. Evaluate on a fresh held-out window
  5. Model gate: only promote to production if new MAPE is within
     5% of baseline (degradation gate prevents silently bad models)
  6. Log everything to MLflow — full audit trail

Designed for production-grade reliability:
  - Creates logs/retrain.log
  - Saves mlruns entry for each run with promotion outcome
  - Exits with code 1 on gate failure so CI/CD can catch it
"""

import sys
import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT       = Path(__file__).parent.parent
LOGS_DIR   = ROOT / "logs"
DATA_PATH  = ROOT / "data" / "demand.csv"
MODEL_PATH = ROOT / "data" / "xgb_model.pkl"
STATE_PATH = ROOT / "data" / "retrain_state.json"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "retrain.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

MAPE_DEGRADATION_GATE = 0.05   # allow up to 5% relative degradation
BASELINE_MAPE         = 1.52   # from original training run


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_run": None, "last_end_date": "2024-01-01", "baseline_mape": BASELINE_MAPE}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def fetch_new_data(start_date: str):
    """Fetch EIA data since start_date.  Falls back to existing CSV slice."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        log.warning("EIA_API_KEY not set — using existing CSV data")
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
            return df[df["timestamp"] > start_date]
        return pd.DataFrame()

    try:
        import requests
        url    = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        params = {
            "api_key":    api_key,
            "frequency":  "hourly",
            "data[0]":    "value",
            "facets[respondent][]": "PJM",
            "facets[type][]":       "D",
            "start":      start_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("response", {}).get("data", [])

        rows = []
        for r in records:
            ts  = pd.to_datetime(str(r["period"]).replace("T", " ").split("+")[0])
            val = r.get("value")
            if val is not None:
                rows.append({"timestamp": ts, "demand_gw": int(val) / 1000.0})

        df = pd.DataFrame(rows)
        log.info(f"Fetched {len(df)} new hourly rows from EIA")
        return df

    except Exception as exc:
        log.error(f"EIA fetch failed: {exc}  — falling back to CSV")
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
            return df[df["timestamp"] > start_date]
        return pd.DataFrame()


def build_and_train(df: pd.DataFrame):
    from src.features import build_features, train_test_split_temporal

    feat_df = build_features(df)
    train_df, test_df = train_test_split_temporal(feat_df)

    exclude      = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    model = XGBRegressor(
        n_estimators=700, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(train_df[feature_cols].values, train_df["demand_gw"].values)
    preds    = model.predict(test_df[feature_cols].values)
    mape_val = mape(test_df["demand_gw"].values, preds)

    return model, feature_cols, mape_val


def retrain():
    log.info("=" * 55)
    log.info("Starting automated retraining pipeline")
    log.info("=" * 55)

    state        = load_state()
    baseline_m   = state.get("baseline_mape", BASELINE_MAPE)
    last_end     = state.get("last_end_date", "2024-01-01")
    log.info(f"Baseline MAPE: {baseline_m:.2f}%  |  Fetching data since {last_end}")

    # Merge new data with existing
    existing_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]) if DATA_PATH.exists() else pd.DataFrame()
    new_df      = fetch_new_data(last_end)

    if new_df.empty and existing_df.empty:
        log.error("No data available for retraining — aborting")
        sys.exit(1)

    if new_df.empty:
        log.warning("No new rows fetched — retraining on existing data")
        combined = existing_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates("timestamp")

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    new_end  = str(combined["timestamp"].max().date())

    log.info(f"Combined dataset: {len(combined):,} rows  |  up to {new_end}")

    model, feature_cols, new_mape = build_and_train(combined)
    log.info(f"New model MAPE:  {new_mape:.2f}%  |  Baseline: {baseline_m:.2f}%")

    gate_limit   = baseline_m * (1 + MAPE_DEGRADATION_GATE)
    gate_passed  = new_mape <= gate_limit
    outcome      = "PROMOTED" if gate_passed else "REJECTED"

    log.info(f"Gate limit: {gate_limit:.2f}%  |  Outcome: {outcome}")

    # Log to MLflow
    try:
        from src.tracker import ExperimentTracker
        tracker = ExperimentTracker("energy-demand-forecast", ROOT / "mlruns")
        tracker.log_run(
            run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}",
            params={"data_rows": len(combined), "new_end_date": new_end,
                    "gate_pct": MAPE_DEGRADATION_GATE},
            metrics={"new_mape": new_mape, "baseline_mape": baseline_m,
                     "gate_limit": gate_limit},
            tags={"outcome": outcome, "model": "XGBoost"},
        )
    except Exception as e:
        log.warning(f"MLflow log failed: {e}")

    if gate_passed:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)
        log.info(f"Model promoted to production  →  {MODEL_PATH}")
        state.update({"last_run": datetime.now().isoformat(),
                      "last_end_date": new_end,
                      "baseline_mape": new_mape})
        save_state(state)
    else:
        log.warning(
            f"Model REJECTED — MAPE {new_mape:.2f}% exceeds gate {gate_limit:.2f}%. "
            "Previous model retained."
        )
        sys.exit(1)

    log.info("Retraining pipeline complete")
    return new_mape


if __name__ == "__main__":
    retrain()
