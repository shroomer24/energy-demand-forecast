"""
scripts/export_models.py
------------------------
Run this after training to copy model artifacts from data/ → models/.
The models/ directory is tracked in git so Render has the files on deploy.

Usage (from project root):
    python3 scripts/export_models.py

What gets copied:
    data/xgb_model.pkl        → models/xgb_model.pkl
    data/interval_models.pkl  → models/interval_models.pkl
    data/demand_seed.csv      → models/demand_seed.csv   (last 200 rows)
    data/shap_importance.png  → models/shap_importance.png
    data/shap_summary.png     → models/shap_summary.png

After running:
    git add models/
    git commit -m "Update trained models"
    git push origin main
    → Render auto-deploys within ~2 minutes
"""

import shutil
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
MODELS = ROOT / "models"

MODELS.mkdir(exist_ok=True)

# Files to copy directly
DIRECT = [
    "xgb_model.pkl",
    "interval_models.pkl",
    "shap_importance.png",
    "shap_summary.png",
]

copied, skipped = [], []

for fname in DIRECT:
    src = DATA / fname
    if src.exists():
        dst = MODELS / fname
        shutil.copy2(src, dst)
        size_kb = dst.stat().st_size / 1024
        copied.append(f"  {fname} ({size_kb:.0f} KB)")
    else:
        skipped.append(f"  {fname} — not found (run the training script first)")

# Export demand seed (last 200 rows) — small file, safe to track in git
demand_src = DATA / "demand.csv"
if demand_src.exists():
    df   = pd.read_csv(demand_src, parse_dates=["timestamp"])
    seed = df.sort_values("timestamp").tail(200)[["timestamp", "demand_gw"]]
    out  = MODELS / "demand_seed.csv"
    seed.to_csv(out, index=False)
    size_kb = out.stat().st_size / 1024
    copied.append(f"  demand_seed.csv ({size_kb:.0f} KB, {len(seed)} rows)")
else:
    skipped.append("  demand_seed.csv — demand.csv not found (run data pipeline first)")

print("\nExported:")
for m in copied:
    print(m)

if skipped:
    print("\nSkipped:")
    for m in skipped:
        print(m)

print(f"\nModels directory: {MODELS}")
print("\nNext steps:")
print("  git add models/")
print('  git commit -m "Add trained models for Render deployment"')
print("  git push origin main")
