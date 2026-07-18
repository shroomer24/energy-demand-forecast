"""
src/weather.py
--------------
Fetches real hourly temperature data from Open-Meteo — free, no API key required.

Historical API:  https://archive-api.open-meteo.com/v1/archive
Forecast API:    https://api.open-meteo.com/v1/forecast

Location: Philadelphia, PA (lat=39.95, lon=-75.16)
Chosen as the geographic centroid of the PJM Interconnection grid footprint,
which spans from New Jersey to Illinois.

Temperature is returned in Fahrenheit and aligned to UTC hourly timestamps
to match the EIA demand data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# Philadelphia, PA — centroid of PJM grid footprint
LAT = 39.95
LON = -75.16

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_historical_temperature(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly temperature for the PJM region from Open-Meteo archive.

    Open-Meteo archive typically lags ~5 days behind present.
    For recent days, use fetch_forecast_temperature() instead.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"  (inclusive)

    Returns:
        DataFrame with columns:
            timestamp     — UTC datetime (hourly)
            temperature_f — temperature in Fahrenheit
    """
    print(f"[weather] Fetching historical temperature {start_date} → {end_date} ...")

    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude":          LAT,
            "longitude":         LON,
            "start_date":        start_date,
            "end_date":          end_date,
            "hourly":            "temperature_2m",
            "timezone":          "UTC",
            "temperature_unit":  "fahrenheit",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "timestamp":     pd.to_datetime(data["hourly"]["time"]),
        "temperature_f": data["hourly"]["temperature_2m"],
    })
    df["temperature_f"] = df["temperature_f"].astype(float)
    print(f"[weather] Got {len(df):,} hourly temperature records")
    return df


def fetch_forecast_temperature(days_ahead: int = 16) -> pd.DataFrame:
    """
    Fetch hourly temperature forecast from Open-Meteo (up to 16 days ahead).
    Also covers the last 2 days so it bridges the archive lag gap.

    Returns:
        DataFrame with columns:
            timestamp     — UTC datetime (hourly)
            temperature_f — temperature in Fahrenheit
    """
    days_ahead = min(16, max(1, days_ahead))
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude":          LAT,
            "longitude":         LON,
            "hourly":            "temperature_2m",
            "timezone":          "UTC",
            "temperature_unit":  "fahrenheit",
            "forecast_days":     days_ahead,
            "past_days":         2,          # overlap with archive to avoid gaps
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "timestamp":     pd.to_datetime(data["hourly"]["time"]),
        "temperature_f": data["hourly"]["temperature_2m"],
    })
    df["temperature_f"] = df["temperature_f"].astype(float)
    return df


def build_full_temperature_series(start_date: str) -> pd.DataFrame:
    """
    Build a complete hourly temperature series from start_date to now + 16 days.

    Combines:
      1. Open-Meteo archive (start_date → ~5 days ago)
      2. Open-Meteo forecast (last 2 days → +16 days ahead)

    The two series are merged and deduplicated so there are no gaps.

    Returns:
        DataFrame with columns: timestamp (UTC), temperature_f
    """
    # Archive covers up to ~5 days ago
    archive_end = (datetime.utcnow() - timedelta(days=6)).strftime("%Y-%m-%d")
    df_hist = fetch_historical_temperature(start_date, archive_end)

    # Forecast covers last 2 days + next 16 days
    df_fcast = fetch_forecast_temperature(days_ahead=16)

    # Merge, keeping forecast values where they overlap (more up-to-date)
    combined = pd.concat([df_hist, df_fcast])
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


# ── Fallback synthetic temperature (used when Open-Meteo is unavailable) ──────

import numpy as np

def synthetic_temperature(timestamps: pd.Series) -> pd.Series:
    """
    Philadelphia seasonal + diurnal proxy — used only as fallback
    when Open-Meteo is unreachable.
    """
    doy  = timestamps.dt.dayofyear
    hour = timestamps.dt.hour
    return (
        55
        + (-22 * np.cos(2 * np.pi * (doy - 15) / 365))
        + (  6 * np.sin(2 * np.pi * (hour - 6)  / 24))
    ).round(1)


if __name__ == "__main__":
    # Quick smoke-test
    df = build_full_temperature_series("2026-07-01")
    print(df.tail())
    print(f"Total rows: {len(df)}")
