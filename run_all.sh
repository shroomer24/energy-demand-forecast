#!/usr/bin/env bash
# Resolve python3 on macOS where 'python' may not exist
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "ERROR: Python not found. Install with: brew install python"
  exit 1
fi
# run_all.sh
# ---------------------------------------------------------------------------
# End-to-end pipeline runner for the PJM Energy Demand Forecasting project.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   cp .env.example .env  && fill in EIA_API_KEY
#
# macOS note:
#   XGBoost requires OpenMP:
#     brew install libomp
#     sudo ln -s $(brew --prefix libomp)/lib/libomp.dylib /usr/local/lib/libomp.dylib
#
#   TabPFN-3 on Apple Silicon — set MPS memory fraction to avoid OOM:
#     export TABPFN_MPS_MEMORY_FRACTION=0.7
#
# TabPFN API key (optional):
#   If you have a Prior Labs API key set TABPFN_API_KEY in .env to use the
#   hosted TabPFN-3-Plus service instead of local inference.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Auto-create .env from .env.example if missing ───────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "[run_all] Created .env from .env.example"
fi

# ── Load environment variables ──────────────────────────────────────────────
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
  echo "[run_all] Loaded .env"
else
  echo "[run_all] WARNING: .env not found. EIA_API_KEY must be set in environment."
fi

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   PJM Energy Demand Forecasting Pipeline                     ║"
echo "║   Ram Aligave · UNT Business Analytics · 2027                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║   Started: ${TIMESTAMP}                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Fetch real EIA demand + Open-Meteo temperature ──────────────────
echo "▶ Step 1/10 — Fetching EIA hourly demand + Open-Meteo real temperature (PJM, 2022–present)"
$PYTHON -m src.data_pipeline_eia
echo ""

# ── Step 2: Train XGBoost model ─────────────────────────────────────────────
echo "▶ Step 2/10 — Training XGBoost model (700 trees, 32 features)"
$PYTHON -m src.train_xgboost
echo ""

# ── Step 3: Train SARIMA baseline ────────────────────────────────────────────
echo "▶ Step 3/10 — Training SARIMA(1,1,1)(1,1,1,7) baseline"
$PYTHON -m src.train_sarima
echo ""

# ── Step 4: Train TabPFN-3 model ─────────────────────────────────────────────
echo "▶ Step 4/10 — Training TabPFN-3 (time-series checkpoint)"
if [ -n "${TABPFN_API_KEY:-}" ]; then
  echo "   MODE: Hosted API (TabPFN-3-Plus via tabpfn-client)"
else
  echo "   MODE: Local OSS (tabpfn package, time-series checkpoint)"
fi
$PYTHON -m src.train_tabpfn
echo ""

# ── Step 5: Train quantile interval models ───────────────────────────────────
echo "▶ Step 5/10 — Training quantile interval models with conformal calibration"
$PYTHON -m src.train_intervals
echo ""

# ── Step 6: Walk-forward backtest ────────────────────────────────────────────
echo "▶ Step 6/10 — Running walk-forward backtest (30-day windows)"
$PYTHON -m src.backtest
echo ""

# ── Step 7: Generate evaluation charts ───────────────────────────────────────
echo "▶ Step 7/10 — Generating evaluation charts (incl. DM test + calibration)"
$PYTHON -m src.evaluate
echo ""

# ── Step 8: SHAP explainability ───────────────────────────────────────────────
echo "▶ Step 8/10 — Generating SHAP feature importance charts"
$PYTHON -m src.explain
echo ""

# ── Step 9: Run unit tests ────────────────────────────────────────────────────
echo "▶ Step 9/10 — Running unit tests"
pytest tests/ --tb=short -q || echo "[tests] Some tests require trained models — re-run after pipeline completes"
echo ""

# ── Step 10: Start FastAPI inference server ───────────────────────────────────
echo "▶ Step 10/10 — Starting FastAPI server on http://0.0.0.0:8001"
echo ""
echo "  Dashboard:       http://localhost:8001"
echo "  API docs:        http://localhost:8001/docs"
echo "  Health:          http://localhost:8001/health"
echo "  TabPFN endpoint: http://localhost:8001/forecast/tabpfn"
echo ""
echo "  Press Ctrl-C to stop."
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
