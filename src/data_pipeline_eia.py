"""
data_pipeline_eia.py
---------------------
Fetches real hourly electricity demand from the U.S. Energy Information
Administration (EIA) Open Data API v2 — PJM Interconnection region,
and merges in real hourly temperature from Open-Meteo (free, no key needed).

EIA docs:       https://www.eia.gov/opendata/documentation.php
Open-Meteo:     https://open-meteo.com/
Dataset:        electricity/rto/region-data  |  type=D (demand)  |  frequency=hourly
Location:       Philadelphia, PA — centroid of PJM grid footprint

Output: data/demand.csv  (columns: timestamp, demand_gw, temperature_f)
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

# Auto-load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.weather import build_full_temperature_series, synthetic_temperature

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("EIA_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "EIA_API_KEY is not set.\n"
        "Add it to a .env file:  EIA_API_KEY=your_key_here\n"
        "Register free at:       https://www.eia.gov/opendata/"
    )
REGION     = "PJM"
START      = "2022-01-01T00"

# END is dynamic: 2 days before today to account for EIA's ~1-2 day reporting lag.
# PJM hourly demand is updated by EIA in near-real-time; there is no hard cap at
# 2024 — the old hardcoded END was simply never updated. This fetches everything
# available up to 48 hours ago, keeping the dataset current on every run.
_end_dt    = datetime.utcnow() - timedelta(days=2)
END        = _end_dt.strftime("%Y-%m-%dT%H")

PAGE_SIZE  = 5000
BASE_URL   = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OUT_DIR    = Path("data")

# ── Fetch with pagination ─────────────────────────────────────────────────────
def fetch_eia_demand() -> pd.DataFrame:
    records = []
    offset  = 0

    print(f"[eia] Fetching hourly demand for {REGION} from {START} to {END} ...")

    while True:
        params = {
            "api_key":              API_KEY,
            "frequency":            "hourly",
            "data[]":               "value",
            "facets[respondent][]": REGION,
            "facets[type][]":       "D",
            "start":                START,
            "end":                  END,
            "offset":               offset,
            "length":               PAGE_SIZE,
        }

        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        page = payload["response"]["data"]
        if not page:
            break

        records.extend(page)
        total = int(payload["response"].get("total", 0))
        print(f"[eia]   fetched {len(records):,} / {total:,} records", end="\r")

        if len(records) >= int(total):
            break

        offset += PAGE_SIZE
        time.sleep(0.25)   # be polite to the API

    print(f"\n[eia] Download complete — {len(records):,} rows")
    return pd.DataFrame(records)


# ── Clean & format ────────────────────────────────────────────────────────────
def process(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Parse timestamp  (format: "2022-01-01T00")
    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # EIA value is in MWh → convert to GW
    df["demand_gw"] = pd.to_numeric(df["value"], errors="coerce") / 1000.0

    # Drop nulls and duplicates
    df = df.dropna(subset=["demand_gw"])
    df = df.drop_duplicates(subset=["timestamp"])

    # Fill any gaps in the hourly index
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_idx)
    df.index.name = "timestamp"
    df["demand_gw"] = df["demand_gw"].interpolate(method="time")
    df = df.reset_index()

    # ── Real temperature from Open-Meteo ─────────────────────────────────────
    start_str = df["timestamp"].min().strftime("%Y-%m-%d")
    try:
        df_temp = build_full_temperature_series(start_str)
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"]).dt.tz_localize(None)
        df = df.merge(df_temp[["timestamp", "temperature_f"]], on="timestamp", how="left")
        # Fill any remaining gaps (e.g. very recent hours Open-Meteo hasn't published)
        missing = df["temperature_f"].isna().sum()
        if missing > 0:
            print(f"[eia] Filling {missing} missing temperature values with seasonal proxy")
            fallback = synthetic_temperature(df.loc[df["temperature_f"].isna(), "timestamp"])
            df.loc[df["temperature_f"].isna(), "temperature_f"] = fallback.values
        print(f"[weather] Real temperature merged ({df['temperature_f'].notna().sum():,} rows)")
    except Exception as exc:
        print(f"[weather] Open-Meteo unavailable ({exc}) — using seasonal proxy")
        df["temperature_f"] = synthetic_temperature(df["timestamp"])
    return df[["timestamp", "demand_gw", "temperature_f"]]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    df_raw = fetch_eia_demand()
    df     = process(df_raw)

    out = OUT_DIR / "demand.csv"
    df.to_csv(out, index=False)

    print(f"[eia] Saved {len(df):,} rows → {out}")
    print(f"[eia] Demand range: {df['demand_gw'].min():.1f} – {df['demand_gw'].max():.1f} GW")
    print(f"[eia] Date range:   {df['timestamp'].min()} → {df['timestamp'].max()}")
