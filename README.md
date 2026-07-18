# Energy Demand Forecasting Pipeline

**End-to-end ML pipeline for hourly electricity demand forecasting on the PJM Interconnection grid.**

Built by **Ram Aligave** · Business Analytics BBA · University of North Texas · Dec 2027  
Contact: ram.aligave1@gmail.com · (469) 422-0627 · Allen, TX

---

## Project Summary

This project benchmarks XGBoost against SARIMA for short-term energy demand forecasting using three years (2022–2024) of real hourly demand data from the U.S. Energy Information Administration (EIA) API.  Key results:

| Model   | MAE (GW) | RMSE (GW) | MAPE   |
|---------|----------|-----------|--------|
| XGBoost | **0.576**    | **0.751**     | **1.52%** |
| SARIMA  | 2.591    | 2.970     | 7.10%  |

XGBoost achieves **4.7× lower MAPE** by leveraging 26 engineered features including lag windows, Fourier time encodings, and a data-center load growth proxy.

---

## Why This Matters (2026 Context)

AI data centers are adding unprecedented load to the grid.  PJM — the largest electricity market in North America — is managing the fastest demand growth in 20 years.  Accurate short-term forecasting directly impacts:

- **Reserve margin planning** — prevents blackouts during demand spikes
- **Renewable curtailment** — wind and solar generation scheduling
- **Real-time pricing** — LMP (locational marginal price) signals

---

## Architecture

```
EIA API (real data)
      │
      ▼
data_pipeline_eia.py   ─── Fetches 26,000+ hourly rows · PJM 2022-2024
      │
      ▼
features.py            ─── 26 engineered features: lags, Fourier, calendar, temp
      │
      ├──▶ train_xgboost.py    ─── 700-tree XGBoost · MLflow tracking · MAE 0.576 GW
      ├──▶ train_sarima.py     ─── SARIMA(1,1,1)(1,1,1,7) baseline · MAE 2.591 GW
      ├──▶ train_intervals.py  ─── Quantile regression (10th/90th pct) · 80% CI
      ├──▶ backtest.py         ─── Walk-forward validation · 30-day rolling windows
      ├──▶ evaluate.py         ─── 5 evaluation charts
      └──▶ retrain.py          ─── Automated retraining · MAPE gate (5% tolerance)
                                        │
                                        ▼
                               api/main.py          ─── FastAPI · port 8001 · CORS
                                        │
                                        ▼
                               frontend/index.html  ─── Chart.js dashboard · CI band
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ramaligave/energy-forecast.git
cd energy-forecast
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**macOS (Apple Silicon / Intel) — XGBoost requires OpenMP:**
```bash
brew install libomp
sudo ln -s $(brew --prefix libomp)/lib/libomp.dylib /usr/local/lib/libomp.dylib
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set EIA_API_KEY=<your key from https://www.eia.gov/opendata/>
```

### 3. Run the full pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

This executes all 7 steps sequentially then starts the API.

### 4. Open the dashboard

Navigate to [http://localhost:8001](http://localhost:8001)

---

## Individual Steps

```bash
# 1. Fetch data
python -m src.data_pipeline_eia

# 2. Train XGBoost
python -m src.train_xgboost

# 3. Train SARIMA baseline
python -m src.train_sarima

# 4. Train prediction intervals
python -m src.train_intervals

# 5. Walk-forward backtest
python -m src.backtest

# 6. Generate charts
python -m src.evaluate

# 7. Start API (port 8001)
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Automated retraining (run on schedule)
python -m src.retrain
```

---

## API Endpoints

| Method | Endpoint              | Description                                     |
|--------|-----------------------|-------------------------------------------------|
| GET    | `/health`             | Liveness probe                                  |
| GET    | `/model/info`         | Model metadata and feature count                |
| POST   | `/forecast`           | Point forecast for N hours                      |
| POST   | `/forecast/intervals` | Forecast with 80% prediction interval band      |
| GET    | `/forecast/latest`    | Most recent 72-hour interval forecast           |
| GET    | `/docs`               | Interactive Swagger UI                          |

**Example request:**
```bash
curl -X POST http://localhost:8001/forecast/intervals \
  -H "Content-Type: application/json" \
  -d '{"hours": 72}'
```

---

## Features (26 total)

| Category | Features |
|----------|----------|
| Calendar | hour, dayofweek, month, dayofyear, is_weekend, is_holiday |
| Fourier  | hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos |
| Lags     | demand_lag_1h, 2h, 24h, 48h, 168h (1-week) |
| Rolling  | roll_mean_24h, roll_std_24h, roll_mean_168h |
| Temperature | temp_sq, temp_cooling (>65°F), temp_heating (<45°F) |
| Trend    | data_center_load_gw (AI load growth proxy) |

---

## MLflow Tracking

All experiments are logged to `./mlruns/`.

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

Each run records:
- Hyperparameters
- MAE / RMSE / MAPE
- Model artifact
- Training duration

---

## Docker

```bash
docker build -t energy-forecast .
docker run -p 8001:8001 --env-file .env energy-forecast
```

---

## Project Structure

```
energy-forecast/
├── src/
│   ├── data_pipeline_eia.py   # EIA API fetch + preprocessing
│   ├── features.py            # Feature engineering
│   ├── tracker.py             # MLflow experiment tracker
│   ├── train_xgboost.py       # XGBoost training
│   ├── train_sarima.py        # SARIMA training
│   ├── train_intervals.py     # Quantile regression intervals
│   ├── backtest.py            # Walk-forward validation
│   ├── evaluate.py            # Evaluation charts
│   └── retrain.py             # Automated retraining + gate
├── api/
│   └── main.py                # FastAPI inference server
├── frontend/
│   └── index.html             # Chart.js dashboard
├── data/                      # Generated (gitignored)
│   ├── demand.csv
│   ├── xgb_model.pkl
│   ├── interval_models.pkl
│   └── charts/
├── mlruns/                    # MLflow runs (gitignored)
├── logs/                      # Retrain logs (gitignored)
├── run_all.sh
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```

---

## Resume Bullet

> Built an end-to-end hourly electricity demand forecasting pipeline benchmarking SARIMA against XGBoost across 26 engineered features, achieving 1.52% MAPE. Tracked experiments with MLflow and deployed a FastAPI inference endpoint — targeting PJM-style grid load forecasting under data center growth scenarios.

---

## Skills Demonstrated

- **Time series forecasting**: SARIMA, XGBoost with lag features, Fourier encodings
- **Production ML patterns**: walk-forward backtesting, quantile regression, automated retraining with model gate
- **MLOps**: MLflow experiment tracking, artifact logging, model promotion workflow
- **API development**: FastAPI, Pydantic v2, CORS, static file serving
- **Data engineering**: EIA REST API, real-world data cleaning, feature pipelines
- **Visualization**: Matplotlib, Chart.js, confidence interval bands

---

## Data Source

U.S. Energy Information Administration (EIA) Open Data API  
Respondent: PJM Interconnection (largest U.S. electricity market)  
Type: D (Demand)  
Frequency: Hourly  
Period: 2022–2024  
License: Public domain (U.S. government data)

EIA API registration: [https://www.eia.gov/opendata/](https://www.eia.gov/opendata/)
