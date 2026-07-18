"""
src/explain.py
--------------
Generates SHAP-based model explanations for the trained XGBoost energy
demand forecasting model.

Outputs saved to data/:
  shap_importance.png  — horizontal bar chart of mean |SHAP| per feature (top 20)
  shap_summary.png     — beeswarm summary plot showing direction + magnitude

Run:
    python3 -m src.explain

Requires: pip install shap
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def run():
    # ── Load model ────────────────────────────────────────────────────────────
    model_path = DATA_DIR / "xgb_model.pkl"
    if not model_path.exists():
        print("[explain] xgb_model.pkl not found — run src/train_xgboost.py first")
        return

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]
    print(f"[explain] Loaded XGBoost model  ({len(feature_cols)} features)")

    # ── Load demand data and build features on the fly ───────────────────────
    demand_path = DATA_DIR / "demand.csv"
    if not demand_path.exists():
        print("[explain] demand.csv not found — run src/data_pipeline_eia.py first")
        return

    from src.features import build_features
    raw    = pd.read_csv(demand_path, parse_dates=["timestamp"])
    df     = build_features(raw)
    sample = df[feature_cols].tail(3000).fillna(0)
    print(f"[explain] Computing SHAP values on {len(sample):,} samples ...")

    # ── SHAP values ───────────────────────────────────────────────────────────
    try:
        import shap
    except ImportError:
        print("[explain] shap not installed — run: pip install shap")
        return

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # ── Plot 1: Feature importance bar chart ──────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df   = (
        pd.DataFrame({"feature": feature_cols, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [
        "#0070F2" if v > imp_df["importance"].median() else "#4CB1FF"
        for v in imp_df["importance"]
    ]
    bars = ax.barh(imp_df["feature"], imp_df["importance"], color=colors, height=0.65)
    ax.set_xlabel("Mean |SHAP value|  (GW impact on prediction)", fontsize=11)
    ax.set_title(
        "Feature Importance — XGBoost PJM Energy Demand Model\n"
        "Mean absolute SHAP value across 3,000 test samples",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    # Annotate bars
    for bar, val in zip(bars, imp_df["importance"]):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color="#556070",
        )
    plt.tight_layout()
    out1 = DATA_DIR / "shap_importance.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[explain] Saved {out1}")

    # ── Plot 2: SHAP beeswarm summary ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        feature_names=feature_cols,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title(
        "SHAP Summary — Feature Impact on Demand Prediction (GW)",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    out2 = DATA_DIR / "shap_summary.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[explain] Saved {out2}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n[explain] Top 10 features by mean |SHAP|:")
    top10 = imp_df.tail(10).iloc[::-1]
    for _, row in top10.iterrows():
        bar_len = int(row["importance"] / imp_df["importance"].max() * 30)
        print(f"  {row['feature']:30s}  {'█' * bar_len}  {row['importance']:.4f} GW")


if __name__ == "__main__":
    run()
