"""
tests/test_api.py
-----------------
Unit tests for the FastAPI inference endpoints.
Uses FastAPI's TestClient — no running server needed.

Run: pytest tests/test_api.py
Note: requires trained model files in data/ or models/.
      If models are not present, tests are skipped gracefully.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Skip entire module if fastapi test client unavailable ─────────────────────
pytest.importorskip("httpx", reason="httpx required for TestClient (pip install httpx)")


def _get_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_ok():
    client = _get_client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


# ── Model info endpoint ───────────────────────────────────────────────────────

def test_model_info_schema():
    client = _get_client()
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    for field in ["status", "point_model_loaded", "feature_count"]:
        assert field in body, f"Missing field: {field}"


def test_model_info_feature_count():
    client = _get_client()
    r = client.get("/model/info")
    body = r.json()
    if body["point_model_loaded"]:
        assert body["feature_count"] == 32, \
            f"Expected 32 features, got {body['feature_count']}"


# ── Forecast endpoint ─────────────────────────────────────────────────────────

def test_forecast_returns_correct_length():
    client = _get_client()
    r = client.post("/forecast", json={"hours": 24})
    if r.status_code == 503:
        pytest.skip("Model not loaded")
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 24


def test_forecast_demand_values_in_range():
    """PJM demand should stay between 40 GW and 200 GW."""
    client = _get_client()
    r = client.post("/forecast", json={"hours": 72})
    if r.status_code == 503:
        pytest.skip("Model not loaded")
    points = r.json()["points"]
    for p in points:
        assert 40 <= p["demand_gw"] <= 200, \
            f"Demand out of range: {p['demand_gw']} GW at {p['timestamp']}"


def test_forecast_timestamps_are_hourly():
    """Consecutive forecast timestamps must be exactly 1 hour apart."""
    client = _get_client()
    r = client.post("/forecast", json={"hours": 24})
    if r.status_code == 503:
        pytest.skip("Model not loaded")
    times = [pd.Timestamp(p["timestamp"]) for p in r.json()["points"]]
    diffs = [(times[i+1] - times[i]).seconds // 3600 for i in range(len(times)-1)]
    assert all(d == 1 for d in diffs), "Forecast timestamps are not exactly 1 hour apart"


def test_forecast_intervals_has_bounds():
    client = _get_client()
    r = client.post("/forecast/intervals", json={"hours": 24})
    if r.status_code == 503:
        pytest.skip("Model not loaded")
    points = r.json()["points"]
    for p in points:
        if p["lower_gw"] is not None and p["upper_gw"] is not None:
            assert p["lower_gw"] <= p["demand_gw"] <= p["upper_gw"], \
                f"Interval ordering violated: {p['lower_gw']} > {p['demand_gw']} or {p['demand_gw']} > {p['upper_gw']}"


def test_shap_endpoint_returns_image_or_404():
    """SHAP endpoint must return either a PNG or a 404 — nothing else."""
    client = _get_client()
    r = client.get("/explain/shap")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["content-type"] == "image/png"
