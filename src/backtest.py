"""
src/backtest.py
---------------
Walk-forward validation framework for energy demand forecasting.

Instead of a single train/test split, this evaluates the model across
multiple rolling windows — the standard approach used in production
forecasting systems.

Method:
  - Start with 12 months of training data
  - Forecast the next 30 days
  - Roll forward 30 days and repeat
  - Report MAE / RMSE / MAPE per window and overall

This proves the model performs consistently over time, not just on
one lucky test split.

Output:
  data/backtest_results.csv
  data/charts/05_backtest.png
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent.parent
DATA_PATH    = ROOT / "data" / "demand.csv"
CHARTS_DIR   = ROOT / "data" / "charts"
RESULTS_PATH = ROOT / "data" / "backtest_results.csv"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 12
TEST_DAYS    = 30


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def build_features_inline(df: pd.DataFrame) -> pd.DataFrame:
    """Inline feature builder to avoid circular imports during backtest."""
    US_HOLIDAYS = {
        "2022-01-17","2022-05-30","2022-07-04","2022-09-05","2022-11-24","2022-12-26",
        "2023-01-02","2023-05-29","2023-07-04","2023-09-04","2023-11-23","2023-12-25",
        "2024-01-01","2024-05-27","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    }
    feat = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = feat["timestamp"]

    feat["hour"]       = ts.dt.hour
    feat["dayofweek"]  = ts.dt.dayofweek
    feat["month"]      = ts.dt.month
    feat["dayofyear"]  = ts.dt.dayofyear
    feat["is_weekend"] = (feat["dayofweek"] >= 5).astype(int)
    feat["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(US_HOLIDAYS).astype(int)

    feat["hour_sin"]  = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow_sin"]   = np.sin(2 * np.pi * feat["dayofweek"] / 7)
    feat["dow_cos"]   = np.cos(2 * np.pi * feat["dayofweek"] / 7)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    temp_col = "temperature_f" if "temperature_f" in feat.columns else "temp_f"
    if temp_col in feat.columns:
        feat["temp_sq"]      = feat[temp_col] ** 2
        feat["temp_cooling"] = np.maximum(0, feat[temp_col] - 65)
        feat["temp_heating"] = np.maximum(0, 45 - feat[temp_col])

    for lag in [1, 2, 24, 48, 168]:
        feat[f"demand_lag_{lag}h"] = feat["demand_gw"].shift(lag)

    feat["roll_mean_24h"]  = feat["demand_gw"].shift(1).rolling(24).mean()
    feat["roll_std_24h"]   = feat["demand_gw"].shift(1).rolling(24).std()
    feat["roll_mean_168h"] = feat["demand_gw"].shift(1).rolling(168).mean()

    t0 = pd.Timestamp("2022-01-01")
    feat["data_center_load_gw"] = 2.0 + ((ts - t0).dt.days / 365) * 0.75

    return feat.dropna()


def run_backtest():
    print("[backtest] Loading demand data ...")
    raw     = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    feat_df = build_features_inline(raw)

    exclude      = {"demand_gw", "timestamp", "temperature_f", "temp_f"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    start_date  = feat_df["timestamp"].min()
    end_date    = feat_df["timestamp"].max()
    train_start = start_date
    train_end   = train_start + pd.DateOffset(months=TRAIN_MONTHS)
    test_end    = train_end   + pd.Timedelta(days=TEST_DAYS)

    folds   = []
    fold_id = 1

    while test_end <= end_date:
        train_mask = (feat_df["timestamp"] >= train_start) & (feat_df["timestamp"] < train_end)
        test_mask  = (feat_df["timestamp"] >= train_end)   & (feat_df["timestamp"] < test_end)

        train = feat_df[train_mask]
        test  = feat_df[test_mask]

        if len(train) < 500 or len(test) < 24:
            break

        model = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(train[feature_cols].values, train["demand_gw"].values)
        preds = model.predict(test[feature_cols].values)

        mae_v  = float(mean_absolute_error(test["demand_gw"].values, preds))
        rmse_v = float(np.sqrt(mean_squared_error(test["demand_gw"].values, preds)))
        mape_v = mape(test["demand_gw"].values, preds)

        folds.append({
            "fold": fold_id,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str(train_end.date()),
            "test_end":    str(test_end.date()),
            "n_train":     len(train),
            "n_test":      len(test),
            "mae":         round(mae_v, 4),
            "rmse":        round(rmse_v, 4),
            "mape":        round(mape_v, 4),
        })

        print(f"[backtest] Fold {fold_id:2d} | "
              f"{str(train_end.date())} → {str(test_end.date())} | "
              f"MAE={mae_v:.3f} GW | MAPE={mape_v:.2f}%")

        train_end = train_end + pd.Timedelta(days=TEST_DAYS)
        test_end  = test_end  + pd.Timedelta(days=TEST_DAYS)
        fold_id  += 1

    results_df = pd.DataFrame(folds)

    print(f"\n── Walk-Forward Backtest Summary ──────────────")
    print(f"   Folds:      {len(folds)}")
    print(f"   Avg MAE:    {results_df['mae'].mean():.3f} GW")
    print(f"   Avg RMSE:   {results_df['rmse'].mean():.3f} GW")
    print(f"   Avg MAPE:   {results_df['mape'].mean():.2f}%")
    print(f"   MAPE range: {results_df['mape'].min():.2f}% – {results_df['mape'].max():.2f}%")
    print(f"───────────────────────────────────────────────\n")

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[backtest] Saved backtest_results.csv")

    _plot_backtest(results_df)
    return results_df


def _plot_backtest(results_df):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#F8F9FA")

    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    colors = ["#0070F2" if m < 3.5 else "#FF6B35" for m in results_df["mape"]]
    bars = ax.bar(results_df["fold"], results_df["mape"],
                  color=colors, edgecolor="none", width=0.6)
    avg = results_df["mape"].mean()
    ax.axhline(avg, color="#222222", linestyle="--", linewidth=1.2,
               label=f"Avg MAPE: {avg:.2f}%")
    for bar, val in zip(bars, results_df["mape"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("Walk-Forward Validation — MAPE per Fold\n"
                 "XGBoost · PJM Interconnection · 30-day forecast windows",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    ax2.plot(results_df["fold"], results_df["mae"],
             color="#0070F2", marker="o", linewidth=2, markersize=5)
    ax2.fill_between(results_df["fold"], results_df["mae"], alpha=0.15, color="#0070F2")
    ax2.set_xlabel("Fold", fontsize=11)
    ax2.set_ylabel("MAE (GW)", fontsize=11)
    ax2.set_title("Mean Absolute Error per Fold", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.5)
    path = CHARTS_DIR / "05_backtest.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[backtest] Saved data/charts/05_backtest.png")


if __name__ == "__main__":
    run_backtest()
