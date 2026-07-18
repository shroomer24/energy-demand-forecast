"""
src/train_tabpfn.py
--------------------
Trains / evaluates TabPFN-3 on the PJM energy demand forecasting task.

TabPFN-3 uses in-context learning: the training set is passed as context
and predictions happen in a single forward pass — no per-dataset gradient
updates. We use the time-series-specialized checkpoint which was fine-tuned
on synthetic time-series patterns and ranks 2nd on fev-bench.

This script supports two inference backends, selected by TABPFN_USE_API:

  LOCAL (default):
    - Uses the open-source `tabpfn` package with local model weights.
    - Requires: pip install tabpfn  (downloads ~500MB checkpoint on first run)
    - License: non-commercial/research use only (TabPFN-3 License v1.0)
    - Hardware: GPU strongly recommended; MPS works on Apple Silicon.
    - Good for: portfolio demos, research, offline use.

  API (set TABPFN_USE_API=1 in .env):
    - Uses `tabpfn-client` which calls Prior Labs' hosted inference service.
    - Requires: pip install tabpfn-client  +  TABPFN_API_KEY in .env
    - Advantages: no GPU needed, access to distributional outputs (quantiles),
      and faster for cold-start runs since no local checkpoint download.
    - Good for: production pipelines, teams without GPU, or when you want
      the full predictive distribution for uncertainty-aware forecasting.

Key outputs:
  data/tabpfn_predictions.csv    — timestamp, demand_gw, tabpfn_pred
  data/tabpfn_model.pkl          — serialized fitted estimator (local mode only)
  MLflow run logged under "energy-demand-forecast"

Usage:
  python src/train_tabpfn.py
  TABPFN_USE_API=1 python src/train_tabpfn.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.features import build_features, train_test_split_temporal
from src.tracker import ExperimentTracker

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "demand.csv"
PRED_PATH  = ROOT / "data" / "tabpfn_predictions.csv"
MODEL_PATH = ROOT / "data" / "tabpfn_model.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("TABPFN_API_KEY", "").strip()
# Auto-enable API mode when TABPFN_API_KEY is present — no need to set TABPFN_USE_API=1
USE_API = bool(API_KEY) or os.getenv("TABPFN_USE_API", "0").strip() == "1"

# Time-series checkpoint from the TabPFN-3 Hugging Face model card.
# Fine-tuned on synthetic time-series patterns — the right choice for
# lag-feature-based demand forecasting.
TS_CHECKPOINT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"

# On Apple Silicon, cap MPS memory to avoid OOM on larger datasets.
# Set to 0.0 to let the OS manage automatically.
MPS_FRACTION = float(os.getenv("TABPFN_MPS_MEMORY_FRACTION", "0.7"))
if MPS_FRACTION > 0:
    os.environ["TABPFN_MPS_MEMORY_FRACTION"] = str(MPS_FRACTION)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _load_local_model():
    """
    Load TabPFNRegressor from the local OSS package.

    On first run, TabPFN will open a browser window for license acceptance
    and download the model checkpoint (~500MB). Subsequent runs use the
    cached checkpoint from TABPFN_MODEL_CACHE_DIR (default: ~/.cache/tabpfn).

    In headless/CI environments, set:
        TABPFN_NO_BROWSER=1
        TABPFN_TOKEN=<your_prior_labs_token>
    """
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        raise ImportError(
            "tabpfn is not installed.\n"
            "Run:  pip install tabpfn\n"
            "Requires Python 3.10+ and PyTorch 2.5+."
        )

    print(f"[tabpfn] Using local OSS package with time-series checkpoint.")
    print(f"[tabpfn] Checkpoint: {TS_CHECKPOINT}")
    print(f"[tabpfn] Note: first run will prompt for license acceptance and")
    print(f"[tabpfn]       download ~500MB checkpoint to ~/.cache/tabpfn")

    return TabPFNRegressor(
        model_path=TS_CHECKPOINT,
        # Do NOT set ignore_pretraining_limits=True unless you've read the docs;
        # the time-series checkpoint is optimized for the dataset sizes we have.
    )


def _load_api_model():
    """
    Load TabPFNRegressor via the tabpfn-client hosted API.

    Set TABPFN_API_KEY in your .env file. Get a key from:
        https://docs.priorlabs.ai/api-reference/getting-started

    The client exposes the same sklearn-like .fit() / .predict() interface
    but routes inference through Prior Labs' managed service. No GPU needed.
    Data is uploaded to Prior Labs' servers — review their data retention and
    privacy terms before using with sensitive data.
    """
    if not API_KEY:
        raise EnvironmentError(
            "TABPFN_USE_API=1 but TABPFN_API_KEY is not set.\n"
            "Add your Prior Labs API key to .env:  TABPFN_API_KEY=your_key_here\n"
            "Get a key at: https://docs.priorlabs.ai/api-reference/getting-started"
        )
    try:
        from tabpfn_client import TabPFNRegressor as APIRegressor
    except ImportError:
        raise ImportError(
            "tabpfn-client is not installed.\n"
            "Run:  pip install tabpfn-client"
        )

    print(f"[tabpfn] Using Prior Labs hosted API (tabpfn-client).")
    print(f"[tabpfn] Data will be uploaded to Prior Labs' inference service.")

    # tabpfn-client authenticates via TABPFN_API_KEY env var automatically.
    return APIRegressor()


# ── Main training function ────────────────────────────────────────────────────

def train():
    # 1. Load and feature-engineer the demand data
    print(f"[tabpfn] Loading demand data from {DATA_PATH} ...")
    raw     = pd.read_csv(DATA_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    feat_df = build_features(raw)

    train_df, test_df = train_test_split_temporal(feat_df)

    # Exclude non-feature columns — same logic as train_xgboost.py
    exclude      = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["demand_gw"].values
    X_test  = test_df[feature_cols].fillna(0).values
    y_test  = test_df["demand_gw"].values

    print(f"[tabpfn] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[tabpfn] Features: {len(feature_cols)}")

    # 2. Load the appropriate backend
    model = _load_api_model() if USE_API else _load_local_model()

    # 3. Fit — TabPFN's .fit() caches the training context (no gradient updates).
    #    This is typically fast (<1s for our ~20k train rows) because TabPFN only
    #    prepares the KV cache, not a full training loop.
    print(f"\n[tabpfn] Fitting model on {len(X_train):,} training rows ...")
    t0 = datetime.utcnow()
    model.fit(X_train, y_train)
    fit_duration = (datetime.utcnow() - t0).total_seconds()
    print(f"[tabpfn] Fit complete in {fit_duration:.1f}s")

    # 4. Predict — always batch (never row-by-row; ~100x slower per TabPFN docs)
    print(f"[tabpfn] Predicting {len(X_test):,} test rows ...")
    t1     = datetime.utcnow()
    preds  = model.predict(X_test)
    pred_duration = (datetime.utcnow() - t1).total_seconds()
    print(f"[tabpfn] Prediction complete in {pred_duration:.1f}s")

    # 5. Metrics
    mae_val  = float(mean_absolute_error(y_test, preds))
    rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape_val = mape(y_test, preds)

    print(f"\n[tabpfn] ── Results ──────────────────────────────────────")
    print(f"[tabpfn]   MAE  : {mae_val:.3f} GW")
    print(f"[tabpfn]   RMSE : {rmse_val:.3f} GW")
    print(f"[tabpfn]   MAPE : {mape_val:.2f}%")
    print(f"[tabpfn]   (XGBoost baseline: MAE=0.576, RMSE=0.751, MAPE=1.52%)")
    print(f"[tabpfn]   (SARIMA baseline:  MAE=2.591, RMSE=2.970, MAPE=7.10%)")
    print(f"[tabpfn] ────────────────────────────────────────────────────")

    # 6. Save predictions
    pred_df = pd.DataFrame({
        "timestamp":   test_df["timestamp"].values,
        "demand_gw":   y_test,
        "tabpfn_pred": np.round(preds, 4),
    })
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"[tabpfn] Predictions saved → {PRED_PATH}")

    # 7. Save model (local mode only — API model is stateless)
    if not USE_API:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":        model,
                "feature_cols": feature_cols,
                "checkpoint":   TS_CHECKPOINT,
                "mape":         mape_val,
                "mae":          mae_val,
                "rmse":         rmse_val,
            }, f)
        print(f"[tabpfn] Model saved → {MODEL_PATH}")

    # 8. Log to MLflow
    tracker = ExperimentTracker("energy-demand-forecast", ROOT / "mlruns")
    tracker.log_run(
        run_name="tabpfn3_timeseries",
        params={
            "backend":    "api" if USE_API else "local",
            "checkpoint": TS_CHECKPOINT,
            "features":   len(feature_cols),
            "train_rows": len(X_train),
            "test_rows":  len(X_test),
            "fit_s":      round(fit_duration, 2),
            "pred_s":     round(pred_duration, 2),
        },
        metrics={
            "mae":  mae_val,
            "rmse": rmse_val,
            "mape": mape_val,
        },
        tags={
            "model":      "TabPFN-3",
            "checkpoint": "timeseries",
            "backend":    "api" if USE_API else "local",
            "stage":      "evaluation",
        },
    )

    return {
        "model":   "TabPFN-3",
        "mae":     mae_val,
        "rmse":    rmse_val,
        "mape":    mape_val,
        "backend": "api" if USE_API else "local",
    }


if __name__ == "__main__":
    results = train()
    print(f"\n[tabpfn] Done. MAPE = {results['mape']:.2f}%")
