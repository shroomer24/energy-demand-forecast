"""
aicore/export_seed.py
---------------------
Run this ONCE after training to export the last 200 rows of demand.csv
as demand_seed.csv — this is baked into the Docker image so the serving
container has lag feature priming data without needing a live DB.

Usage:
    python3 aicore/export_seed.py
"""

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent.parent
IN_PATH   = ROOT / "data" / "demand.csv"
OUT_PATH  = ROOT / "data" / "demand_seed.csv"

df   = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
seed = df.sort_values("timestamp").tail(200)[["timestamp", "demand_gw"]]
seed.to_csv(OUT_PATH, index=False)
print(f"Exported {len(seed)} rows to {OUT_PATH}")
print(f"Last timestamp: {seed['timestamp'].max()}")
