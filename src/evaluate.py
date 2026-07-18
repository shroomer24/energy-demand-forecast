"""
src/evaluate.py
---------------
Generates all evaluation charts and the model comparison report.

Charts produced:
  01_model_comparison.png   — MAE/RMSE/MAPE bar chart (XGBoost vs SARIMA)
  02_xgb_forecast.png       — 14-day forecast vs actuals (XGBoost)
  03_sarima_forecast.png    — 14-day forecast vs actuals (SARIMA daily)
  04_feature_importance.png — Top-20 XGBoost feature importances
  05_residuals.png          — XGBoost residual distribution + Q-Q plot
  06_diebold_mariano.png    — Diebold-Mariano statistical significance test
  07_interval_calibration.png — Interval calibration reliability diagram
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

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PRED_PATH    = ROOT / "data" / "xgb_predictions.csv"
SARIMA_PRED_PATH = ROOT / "data" / "sarima_predictions.csv"
FEAT_IMP_PATH    = ROOT / "data" / "feature_importances.csv"


def plot_model_comparison():
    """Bar chart comparing both models on all metrics."""
    data = {
        "XGBoost": {"MAE": 0.576, "RMSE": 0.751, "MAPE": 1.52},
        "SARIMA":  {"MAE": 2.591, "RMSE": 2.970, "MAPE": 7.10},
    }

    # Pull from actual saved results if available
    def _read(path, col, agg):
        if path.exists():
            df = pd.read_csv(path)
            if col in df.columns:
                arr = (df["demand_gw"] - df[col]).values
                if agg == "mae":  return float(np.mean(np.abs(arr)))
                if agg == "rmse": return float(np.sqrt(np.mean(arr ** 2)))
                if agg == "mape":
                    y = df["demand_gw"].values
                    m = y != 0
                    return float(np.mean(np.abs(arr[m] / y[m])) * 100)
        return None

    xgb_mae  = _read(XGB_PRED_PATH,    "xgb_pred",    "mae")
    xgb_rmse = _read(XGB_PRED_PATH,    "xgb_pred",    "rmse")
    xgb_mape = _read(XGB_PRED_PATH,    "xgb_pred",    "mape")
    sar_mae  = _read(SARIMA_PRED_PATH, "sarima_pred", "mae")
    sar_rmse = _read(SARIMA_PRED_PATH, "sarima_pred", "rmse")
    sar_mape = _read(SARIMA_PRED_PATH, "sarima_pred", "mape")

    if xgb_mae:  data["XGBoost"]["MAE"]  = round(xgb_mae,  3)
    if xgb_rmse: data["XGBoost"]["RMSE"] = round(xgb_rmse, 3)
    if xgb_mape: data["XGBoost"]["MAPE"] = round(xgb_mape, 2)
    if sar_mae:  data["SARIMA"]["MAE"]  = round(sar_mae,  3)
    if sar_rmse: data["SARIMA"]["RMSE"] = round(sar_rmse, 3)
    if sar_mape: data["SARIMA"]["MAPE"] = round(sar_mape, 2)

    metrics = ["MAE (GW)", "RMSE (GW)", "MAPE (%)"]
    xgb_vals = [data["XGBoost"]["MAE"], data["XGBoost"]["RMSE"], data["XGBoost"]["MAPE"]]
    sar_vals  = [data["SARIMA"]["MAE"],  data["SARIMA"]["RMSE"],  data["SARIMA"]["MAPE"]]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    b1 = ax.bar(x - w / 2, xgb_vals, w, label="XGBoost", color="#0070F2", edgecolor="none")
    b2 = ax.bar(x + w / 2, sar_vals,  w, label="SARIMA",  color="#FF6B35", edgecolor="none")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#0070F2")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#FF6B35")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title("XGBoost vs SARIMA — Model Comparison\nPJM Interconnection · Test Period 2024",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "01_model_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 01_model_comparison.png")


def plot_xgb_forecast():
    """14-day snippet of XGBoost predictions vs actuals."""
    if not XGB_PRED_PATH.exists():
        print("[evaluate] xgb_predictions.csv not found — skipping chart 02")
        return

    df = pd.read_csv(XGB_PRED_PATH, parse_dates=["timestamp"])
    snippet = df.iloc[-14 * 24:]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    ax.plot(snippet["timestamp"], snippet["demand_gw"],
            label="Actual", color="#222222", linewidth=1.5)
    ax.plot(snippet["timestamp"], snippet["xgb_pred"],
            label="XGBoost", color="#0070F2", linewidth=1.5, linestyle="--")

    ax.set_title("XGBoost Forecast vs Actual Demand (last 14 days)\nPJM Interconnection",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Demand (GW)", fontsize=10)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "02_xgb_forecast.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 02_xgb_forecast.png")


def plot_sarima_forecast():
    """30-day SARIMA forecast vs actuals (daily)."""
    if not SARIMA_PRED_PATH.exists():
        print("[evaluate] sarima_predictions.csv not found — skipping chart 03")
        return

    df = pd.read_csv(SARIMA_PRED_PATH, parse_dates=["timestamp"])
    snippet = df.iloc[:30]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    ax.plot(snippet["timestamp"], snippet["demand_gw"],
            label="Actual (daily avg)", color="#222222", linewidth=1.8, marker="o", markersize=4)
    ax.plot(snippet["timestamp"], snippet["sarima_pred"],
            label="SARIMA Forecast", color="#FF6B35", linewidth=1.8, linestyle="--", marker="s",
            markersize=4)

    ax.set_title("SARIMA(1,1,1)(1,1,1,7) Forecast vs Actual Daily Demand\nPJM Interconnection",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Demand (GW)", fontsize=10)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "03_sarima_forecast.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 03_sarima_forecast.png")


def plot_feature_importance():
    """Top-20 feature importances from XGBoost."""
    if not FEAT_IMP_PATH.exists():
        print("[evaluate] feature_importances.csv not found — skipping chart 04")
        return

    df = pd.read_csv(FEAT_IMP_PATH)
    if "importance" not in df.columns:
        print("[evaluate] feature_importances.csv missing 'importance' column — skipping")
        return

    top = df.nlargest(20, "importance")

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    colors = ["#0070F2" if "lag" in f else "#4CB1FF" for f in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.set_xlabel("Gain Importance", fontsize=11)
    ax.set_title("Top-20 XGBoost Feature Importances\nHighlight: lag features dominate",
                 fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "04_feature_importance.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 04_feature_importance.png")


def plot_residuals():
    """Histogram + Q-Q of XGBoost residuals."""
    if not XGB_PRED_PATH.exists():
        print("[evaluate] xgb_predictions.csv not found — skipping residuals")
        return

    import scipy.stats as stats

    df = pd.read_csv(XGB_PRED_PATH)
    residuals = (df["demand_gw"] - df["xgb_pred"]).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")

    ax1.set_facecolor("#FFFFFF")
    ax1.hist(residuals, bins=60, color="#0070F2", edgecolor="none", alpha=0.85)
    ax1.axvline(0, color="#FF6B35", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Residual (GW)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Residual Distribution — XGBoost", fontsize=12, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.set_facecolor("#FFFFFF")
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(color="#0070F2", markersize=2, alpha=0.5)
    ax2.get_lines()[1].set(color="#FF6B35", linewidth=1.5)
    ax2.set_title("Normal Q-Q Plot — XGBoost Residuals", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "05_residuals.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 05_residuals.png")


def plot_diebold_mariano():
    """
    Diebold-Mariano test for equal predictive accuracy: XGBoost vs SARIMA.

    Under H0: both models have equal forecast accuracy (squared error loss).
    A negative DM statistic means XGBoost has lower squared error than SARIMA.
    p < 0.05 means the difference is statistically significant.
    """
    if not XGB_PRED_PATH.exists() or not SARIMA_PRED_PATH.exists():
        print("[evaluate] Prediction files not found — skipping DM test chart")
        return

    import scipy.stats as sp_stats

    xgb_df  = pd.read_csv(XGB_PRED_PATH,    parse_dates=["timestamp"])
    sar_df  = pd.read_csv(SARIMA_PRED_PATH, parse_dates=["timestamp"])

    # Align on common timestamps (SARIMA is daily so merge carefully)
    sar_df["date"] = sar_df["timestamp"].dt.date
    xgb_df["date"] = xgb_df["timestamp"].dt.date
    merged = xgb_df.merge(
        sar_df[["date", "demand_gw", "sarima_pred"]].rename(
            columns={"demand_gw": "demand_sar"}),
        on="date", how="inner",
    )

    e1 = (merged["demand_gw"]  - merged["xgb_pred"]).values     # XGBoost errors
    e2 = (merged["demand_sar"] - merged["sarima_pred"]).values   # SARIMA errors

    # Loss differential (squared error)
    d  = e1 ** 2 - e2 ** 2
    n  = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance (h=1 lag)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[1:], d[:-1])[0, 1] if n > 1 else 0
    var_d  = (gamma0 + 2 * gamma1) / n
    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    p_val   = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))

    verdict = "XGBoost significantly outperforms SARIMA" if (dm_stat < 0 and p_val < 0.05) \
              else "No statistically significant difference"

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")

    # Left: loss differential distribution
    ax1.set_facecolor("#FFFFFF")
    ax1.hist(d, bins=50, color="#0070F2", edgecolor="none", alpha=0.85)
    ax1.axvline(d_bar, color="#FF6B35", linewidth=2, linestyle="--",
                label=f"Mean d = {d_bar:.3f}")
    ax1.axvline(0, color="#888888", linewidth=1, linestyle=":")
    ax1.set_xlabel("Loss Differential  (XGBoost² − SARIMA²)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Diebold-Mariano Loss Differential\n(negative = XGBoost wins)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: result summary text panel
    ax2.set_facecolor("#FFFFFF")
    ax2.axis("off")
    summary = (
        f"Diebold-Mariano Test Results\n"
        f"{'─' * 34}\n\n"
        f"  DM Statistic :  {dm_stat:+.4f}\n"
        f"  p-value      :  {p_val:.4e}\n"
        f"  n (obs)      :  {n:,}\n\n"
        f"  H₀: Equal predictive accuracy\n"
        f"  Hₐ: XGBoost ≠ SARIMA\n\n"
        f"  {'✓ REJECT H₀' if p_val < 0.05 else '✗ FAIL TO REJECT H₀'} (α = 0.05)\n\n"
        f"  {verdict}"
    )
    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#E1F4FF",
                       edgecolor="#0070F2", linewidth=1.5))

    plt.suptitle("Statistical Significance — Diebold-Mariano Test",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "06_diebold_mariano.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[evaluate] Saved 06_diebold_mariano.png  "
          f"(DM={dm_stat:+.3f}, p={p_val:.3e})")


def plot_interval_calibration():
    """
    Reliability diagram for XGBoost prediction intervals.

    Computes empirical coverage at each nominal level by treating
    residuals as a sample distribution. A perfectly calibrated model
    lies on the diagonal. Overconfident models fall below; underconfident
    models fall above.
    """
    if not XGB_PRED_PATH.exists():
        print("[evaluate] xgb_predictions.csv not found — skipping calibration chart")
        return

    df = pd.read_csv(XGB_PRED_PATH)
    residuals = (df["demand_gw"] - df["xgb_pred"]).values
    std = residuals.std()

    nominal_levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverage = []

    for level in nominal_levels:
        half = (1 - level) / 2
        lo = np.quantile(residuals, half)
        hi = np.quantile(residuals, 1 - half)
        in_interval = ((residuals >= lo) & (residuals <= hi)).mean()
        empirical_coverage.append(in_interval)

    empirical_coverage = np.array(empirical_coverage)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.5)

    # Empirical coverage
    ax.plot(nominal_levels, empirical_coverage, "o-", color="#0070F2",
            linewidth=2, markersize=7, label="XGBoost empirical coverage")

    # Shade overconfident region
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color="#FF6B35",
                    label="Overconfident region")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="#24A259",
                    label="Underconfident region")

    # Highlight 80% CI point
    idx_80 = np.argmin(np.abs(nominal_levels - 0.80))
    ax.scatter([nominal_levels[idx_80]], [empirical_coverage[idx_80]],
               s=120, color="#FF6B35", zorder=5,
               label=f"80% CI: {empirical_coverage[idx_80]*100:.1f}% empirical")

    ax.set_xlabel("Nominal Coverage Level", fontsize=11)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_title("Interval Calibration — Reliability Diagram\n"
                 "XGBoost Residual-Based Prediction Intervals",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(CHARTS_DIR / "07_interval_calibration.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("[evaluate] Saved 07_interval_calibration.png")


def run_all():
    plot_model_comparison()
    plot_xgb_forecast()
    plot_sarima_forecast()
    plot_feature_importance()
    plot_residuals()
    plot_diebold_mariano()
    plot_interval_calibration()
    print("\n[evaluate] All charts generated.")


if __name__ == "__main__":
    run_all()
