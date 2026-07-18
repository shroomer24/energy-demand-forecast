# ─────────────────────────────────────────────────────────────
# Dockerfile for SAP AI Core serving deployment
# ─────────────────────────────────────────────────────────────
# Build:
#   docker build -f aicore/Dockerfile -t rali24/energy-forecast-server:v1 .
#
# Push:
#   docker push rali24/energy-forecast-server:v1
#
# The model files (xgb_model.pkl, demand_seed.csv) are baked in
# from your local data/ directory at build time.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY aicore/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving code
COPY aicore/serve.py ./serve.py

# Copy trained model and seed data
# These must exist locally before building:
#   data/xgb_model.pkl        — output of src/train_xgboost.py
#   data/demand_seed.csv      — last 200 rows of demand.csv for lag priming
COPY data/xgb_model.pkl      ./models/xgb_model.pkl
COPY data/demand_seed.csv    ./models/demand_seed.csv

# AI Core requires port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
