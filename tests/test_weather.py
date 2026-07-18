"""
tests/test_weather.py
---------------------
Unit tests for src/weather.py — temperature synthesis and API response shape.
Run: pytest tests/test_weather.py
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.weather import synthetic_temperature


def test_synthetic_temperature_length():
    ts = pd.Series(pd.date_range("2023-01-01", periods=8760, freq="h"))
    result = synthetic_temperature(ts)
    assert len(result) == 8760


def test_synthetic_temperature_realistic_range():
    """Philadelphia temps should stay between -20°F and 115°F."""
    ts = pd.Series(pd.date_range("2022-01-01", periods=26280, freq="h"))
    temps = synthetic_temperature(ts)
    assert temps.min() > -20, f"Unrealistically cold: {temps.min():.1f}°F"
    assert temps.max() < 115, f"Unrealistically hot: {temps.max():.1f}°F"


def test_synthetic_temperature_seasonal_signal():
    """January average must be colder than July average."""
    jan = pd.Series(pd.date_range("2023-01-01", periods=744, freq="h"))
    jul = pd.Series(pd.date_range("2023-07-01", periods=744, freq="h"))
    assert synthetic_temperature(jan).mean() < synthetic_temperature(jul).mean(), \
        "January should be colder than July"


def test_synthetic_temperature_diurnal_signal():
    """Midday (12:00) should be warmer than overnight (03:00) on average."""
    days = pd.date_range("2023-06-01", periods=30, freq="D")
    noon     = pd.Series([d + pd.Timedelta(hours=12) for d in days])
    midnight = pd.Series([d + pd.Timedelta(hours=3)  for d in days])
    assert synthetic_temperature(noon).mean() > synthetic_temperature(midnight).mean(), \
        "Midday should be warmer than 3 AM"


def test_synthetic_temperature_no_nans():
    ts = pd.Series(pd.date_range("2022-01-01", periods=1000, freq="h"))
    result = synthetic_temperature(ts)
    assert result.isnull().sum() == 0


def test_synthetic_temperature_returns_series():
    ts = pd.Series(pd.date_range("2023-01-01", periods=24, freq="h"))
    result = synthetic_temperature(ts)
    assert isinstance(result, pd.Series)
