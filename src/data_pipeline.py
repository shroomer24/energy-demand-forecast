"""
data_pipeline.py
----------------
Generates realistic PJM-style hourly electricity demand data (2022-2024).

Demand is modeled from first principles:
  - Base industrial load
  - Residential/commercial daily cycle
  - Temperature-driven HVAC load (cooling + heating)
  - Data center baseline growth (reflects 2022-2024 AI infrastructure buildout)
  - Weekly cycle (weekday vs weekend)
  - Annual seasonality
  - Federal holiday suppression
  - Gaussian noise

Output: data/demand.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── US federal holidays (simplified, fixed dates) ────────────────────────────
FEDERAL_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (12, 25), # Christmas
    (11, 11), # Veterans Day
    (12, 31), # New Year's Eve (partial suppression)
}

def _is_holiday(dt: pd.Timestamp) -> bool:
    return (dt.month, dt.day) in FEDERAL_HOLIDAYS


def _temperature_profile(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Synthetic temperature (°F) for a mid-Atlantic US region (PJM territory).
    Annual sine wave centered on July, ±25°F amplitude around 58°F mean.
    """
    day_of_year = timestamps.dayofyear
    temp_mean = 58.0
    temp_amplitude = 25.0
    # Peak in ~July (day ~196)
    temp = temp_mean + temp_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Add daily variation (warmer afternoons)
    hour = timestamps.hour
    daily_variation = 8.0 * np.sin(np.pi * (hour - 6) / 12)
    daily_variation = np.where((hour >= 6) & (hour <= 18), daily_variation, 0)
    # Add weather noise
    noise = rng.normal(0, 3, len(timestamps))
    return temp + daily_variation + noise


def _hvac_load(temp: np.ndarray) -> np.ndarray:
    """
    HVAC electricity demand (GW) as a function of temperature.
    Cooling demand kicks in above 65°F, heating below 45°F.
    """
    cooling = np.maximum(0, (temp - 65) * 0.18)   # ~0.18 GW per °F above 65
    heating = np.maximum(0, (45 - temp) * 0.12)   # ~0.12 GW per °F below 45
    return cooling + heating


def _daily_load_curve(hour: np.ndarray) -> np.ndarray:
    """
    Normalised residential/commercial load curve (0–1).
    Morning ramp 6–9 AM, midday plateau, evening peak 6–9 PM.
    """
    curve = np.zeros(len(hour))
    for i, h in enumerate(hour):
        if 0 <= h < 6:
            curve[i] = 0.55
        elif 6 <= h < 9:
            curve[i] = 0.55 + (h - 6) / 3 * 0.30   # morning ramp
        elif 9 <= h < 17:
            curve[i] = 0.85
        elif 17 <= h < 21:
            curve[i] = 0.85 + (h - 17) / 4 * 0.15  # evening peak
        else:
            curve[i] = 1.0 - (h - 21) / 3 * 0.45   # evening decline
    return curve


def generate_demand(
    start: str = "2022-01-01",
    end: str = "2024-12-31 23:00",
    output_path: str = "data/demand.csv",
) -> pd.DataFrame:
    """
    Generate hourly electricity demand DataFrame and save to CSV.

    Columns
    -------
    timestamp   : UTC hourly index
    demand_gw   : total electricity demand in GW
    temperature : synthetic temperature in °F
    """
    timestamps = pd.date_range(start=start, end=end, freq="h")
    n = len(timestamps)

    # 1. Base industrial load (relatively flat, slight growth trend)
    base_load = 25.0  # GW
    # Data center growth: +8% annually compounded (AI buildout)
    years_elapsed = (timestamps - pd.Timestamp(start)).total_seconds() / (365.25 * 24 * 3600)
    datacenter_growth = base_load * 0.15 * (1.08 ** years_elapsed - 1)  # starts small, grows

    # 2. Temperature-driven HVAC
    temp = _temperature_profile(timestamps)
    hvac = _hvac_load(temp)

    # 3. Daily residential/commercial cycle
    daily_curve = _daily_load_curve(timestamps.hour.values)
    residential_peak = 18.0  # GW max residential+commercial
    residential = residential_peak * daily_curve

    # 4. Weekend reduction (industrial + commercial down ~15%)
    is_weekend = timestamps.dayofweek >= 5
    weekend_factor = np.where(is_weekend, 0.85, 1.0)

    # 5. Holiday suppression
    holiday_factor = np.array([0.78 if _is_holiday(ts) else 1.0 for ts in timestamps])

    # 6. Combine
    demand = (
        (base_load + datacenter_growth + residential + hvac)
        * weekend_factor
        * holiday_factor
    )

    # 7. Add realistic noise (measurement + micro-events)
    noise = rng.normal(0, 0.6, n)
    demand = np.maximum(demand + noise, 10.0)  # floor at 10 GW

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_gw": np.round(demand, 3),
        "temperature_f": np.round(temp, 1),
    })
    df.set_index("timestamp", inplace=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"[data_pipeline] Saved {len(df):,} rows → {output_path}")
    print(f"[data_pipeline] Demand range: {df.demand_gw.min():.1f} – {df.demand_gw.max():.1f} GW")
    print(f"[data_pipeline] Date range:   {df.index[0]} → {df.index[-1]}")
    return df


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    generate_demand()
