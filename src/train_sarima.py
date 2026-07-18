"""
src/train_sarima.py
-------------------
Trains a SARIMA(1,1,1)(1,1,1,24) model as the statistical baseline.

SARIMA captures trend and both daily and weekly seasonality through
differencing — a strong benchmark for the ML models to beat.

MLflow logs:
  - model order parameters
  - AIC / BIC
  - MAE, RMSE, MAPE on held-out test period
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.tracker import ExperimentTracker

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "demand.csv"
PRED_PATH = ROOT / "data" / "sarima_predictions.csv"
TEST_DATE = "2024-10-01"


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train():
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.set_index("timestamp").asfreq("h")

    # Downsample to daily for SARIMA (hourly is too slow)
    daily = df["demand_gw"].resample("D").mean()

    train = daily[daily.index < TEST_DATE]
    test  = daily[daily.index >= TEST_DATE]

    print(f"[sarima] Training on {len(train)} days, testing on {len(test)} days")
    print(f"[sarima] Fitting model (this may take ~30s)...")

    import time
    t0    = time.time()
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False)
    res   = model.fit(disp=False)
    duration = time.time() - t0

    preds    = res.forecast(steps=len(test))
    mae_val  = float(mean_absolute_error(test.values, preds.values))
    rmse_val = float(np.sqrt(mean_squared_error(test.values, preds.values)))
    mape_val = mape(test.values, preds.values)

    print(f"[sarima] MAE={mae_val:.3f} GW  |  RMSE={rmse_val:.3f} GW  |  MAPE={mape_val:.2f}%")

    # Save predictions
    pred_df = pd.DataFrame({
        "timestamp": test.index,
        "demand_gw": test.values,
        "sarima_pred": preds.values,
    })
    pred_df.to_csv(PRED_PATH, index=False)

    # Log
    tracker = ExperimentTracker("energy-demand-forecast", ROOT / "mlruns")
    tracker.log_run(
        run_name="sarima_baseline",
        params={"order": "(1,1,1)", "seasonal_order": "(1,1,1,7)",
                "frequency": "daily", "duration_s": round(duration, 2)},
        metrics={"mae": mae_val, "rmse": rmse_val, "mape": mape_val,
                 "aic": round(res.aic, 2), "bic": round(res.bic, 2)},
        tags={"model": "SARIMA", "stage": "baseline"},
    )

    return {"model": "SARIMA", "mae": mae_val, "rmse": rmse_val, "mape": mape_val}


if __name__ == "__main__":
    train()
