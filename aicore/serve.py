"""
aicore/serve.py
---------------
FastAPI inference server for SAP AI Core deployment.

Differences from api/main.py:
  - Listens on port 8080 (AI Core requirement)
  - Loads model from /app/models/ (baked into Docker image)
  - Adds /v2/health/ready and /v2/health/live probes (KServe standard)
  - No static frontend — pure inference API
  - No TabPFN or interval models — XGBoost point forecast only

Run locally:
  uvicorn aicore.serve:app --host 0.0.0.0 --port 8080

AI Core starts this automatically via the Dockerfile CMD.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Model is baked into the Docker image at /app/models/
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/xgb_model.pkl"))
DATA_PATH  = Path(os.getenv("DATA_PATH",  "/app/models/demand_seed.csv"))

# ---------- Pydantic models ----------

class ForecastRequest(BaseModel):
    hours: int = 72
    start_timestamp: Optional[str] = None  # ISO format; defaults to now

class ForecastPoint(BaseModel):
    timestamp: str
    demand_gw: float

class ForecastResponse(BaseModel):
    model: str
    generated_at: str
    hours_requested: int
    points: List[ForecastPoint]

class HealthResponse(BaseModel):
    status: str

# ---------- Global state ----------

_state: dict = {
    "model":        None,
    "feature_cols": [],
    "last_ts":      None,
    "seed":         None,
}

US_HOLIDAYS = {
    "2022-01-17","2022-02-21","2022-05-30","2022-07-04",
    "2022-09-05","2022-11-24","2022-12-26",
    "2023-01-02","2023-01-16","2023-02-20","2023-05-29","2023-07-04",
    "2023-09-04","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27",
    "2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-05-26",
    "2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-05-25",
    "2026-07-04","2026-09-07","2026-11-26","2026-12-25",
}


def _load_models():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
            _state["model"]        = obj.get("model")
            _state["feature_cols"] = obj.get("feature_cols", [])
        print(f"[serve] XGBoost model loaded from {MODEL_PATH}")
    else:
        print(f"[serve] WARNING: model not found at {MODEL_PATH}")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        _state["last_ts"] = df["timestamp"].max()
        _state["seed"]    = df.sort_values("timestamp").tail(200)["demand_gw"].reset_index(drop=True)
        print(f"[serve] Seed data loaded — last timestamp: {_state['last_ts']}")
    else:
        print(f"[serve] No seed data found at {DATA_PATH}. Using synthetic seed.")
        _state["seed"] = pd.Series([50.0] * 200)
        _state["last_ts"] = pd.Timestamp.utcnow().floor("h")


def _build_feature_row(ts: pd.Timestamp, demand_history: pd.Series, idx: int) -> dict:
    """Build one feature row — all 32 features matching features.py exactly."""
    date_str = ts.strftime("%Y-%m-%d")
    doy      = ts.day_of_year
    hour     = ts.hour
    temp     = 55 + (-22 * np.cos(2 * np.pi * (doy - 15) / 365)) + \
                    (  6 * np.sin(2 * np.pi * (hour - 6) / 24))

    row = {
        "hour":           ts.hour,
        "dayofweek":      ts.dayofweek,
        "month":          ts.month,
        "dayofyear":      ts.day_of_year,
        "weekofyear":     int(ts.isocalendar()[1]),
        "quarter":        ts.quarter,
        "is_weekend":     int(ts.dayofweek >= 5),
        "is_holiday":     int(date_str in US_HOLIDAYS),
        "hour_sin":       np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":       np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin":        np.sin(2 * np.pi * ts.dayofweek / 7),
        "dow_cos":        np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin":      np.sin(2 * np.pi * ts.month / 12),
        "month_cos":      np.cos(2 * np.pi * ts.month / 12),
        "dayofyear_sin":  np.sin(2 * np.pi * ts.day_of_year / 365),
        "dayofyear_cos":  np.cos(2 * np.pi * ts.day_of_year / 365),
        "temp_sq":        temp ** 2,
        "temp_cooling":   float(max(0, temp - 65)),
        "temp_heating":   float(max(0, 45  - temp)),
    }

    def _lag(lag):
        i = idx - lag
        return float(demand_history.iloc[i]) if i >= 0 else float(demand_history.iloc[0])

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        row[f"demand_lag_{lag}h"] = _lag(lag)

    window24  = demand_history.iloc[max(0, idx - 24):idx]
    window168 = demand_history.iloc[max(0, idx - 168):idx]
    row["demand_roll_mean_24h"]  = float(window24.mean())  if len(window24)  > 0 else 0.0
    row["demand_roll_std_24h"]   = float(window24.std())   if len(window24)  > 1 else 0.0
    row["demand_roll_mean_168h"] = float(window168.mean()) if len(window168) > 0 else 0.0

    t0 = pd.Timestamp("2022-01-01")
    row["data_center_load_gw"] = 2.0 + ((ts - t0).days / 365) * 0.75

    return row


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="Energy Demand Forecasting — AI Core",
    description="XGBoost hourly PJM electricity demand forecast, deployed on SAP AI Core.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Health probes (KServe standard) ----------

@app.get("/v2/health/live", response_model=HealthResponse)
@app.get("/health")
def health_live():
    return {"status": "ok"}


@app.get("/v2/health/ready", response_model=HealthResponse)
def health_ready():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


# ---------- Model info ----------

@app.get("/v2/models/energy-forecast")
def model_info():
    return {
        "name":           "energy-forecast",
        "version":        "1.0.0",
        "framework":      "xgboost",
        "feature_count":  len(_state["feature_cols"]),
        "model_loaded":   _state["model"] is not None,
        "last_seed_ts":   str(_state["last_ts"]) if _state["last_ts"] else None,
    }


# ---------- Forecast endpoint ----------

@app.post("/v2/predict", response_model=ForecastResponse)
@app.post("/forecast")
def forecast(req: ForecastRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_ts = (
        pd.Timestamp(req.start_timestamp)
        if req.start_timestamp
        else (_state["last_ts"] + timedelta(hours=1)
              if _state["last_ts"] is not None
              else pd.Timestamp.utcnow().floor("h"))
    )

    model        = _state["model"]
    feature_cols = _state["feature_cols"]
    seed         = _state["seed"].copy()
    demand_history = seed

    results = []
    for h in range(req.hours):
        ts  = start_ts + timedelta(hours=h)
        idx = len(demand_history) + h
        row = _build_feature_row(
            ts,
            pd.concat([demand_history, pd.Series([0.0] * h)], ignore_index=True),
            idx,
        )
        X    = pd.DataFrame([row])[feature_cols].fillna(0) if feature_cols else pd.DataFrame([row]).fillna(0)
        pred = float(model.predict(X.values)[0])
        results.append({"timestamp": ts.isoformat(), "demand_gw": round(pred, 3)})

    return ForecastResponse(
        model="xgboost-v1",
        generated_at=datetime.utcnow().isoformat(),
        hours_requested=req.hours,
        points=results,
    )
