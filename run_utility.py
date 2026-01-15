# run_utility.py
# Rolling Mean–Variance Utility optimization
#
# pip install -r requirements.txt

from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.data import download_monthly_returns, prepare_returns
from src.optimizers import opt_mean_variance_utility
from src.backtest import rolling_optimize, apply_transaction_costs
from src.reporting import perf_summary, tail_metrics, export_excel


# -----------------------------
# 1) Universe
# -----------------------------
TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "BIL",
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GLD",
}

START = "2005-01-01"
END = None
LOOKBACK_MONTHS = 120
MIN_COVERAGE = 1.0

# -----------------------------
# 2) Bounds
# -----------------------------
BOUNDS = {
    "SP500":     (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "T_BILLS":   (0.00, 0.30),
    "CHINA":     (0.00, 0.30),
    "REITS":     (0.00, 0.30),
    "GOLD":      (0.00, 0.30),
}
DEFAULT_BOUNDS = (0.0, 1.0)

# -----------------------------
# 3) Utility parameter
# -----------------------------
RISK_AVERSION = 5.0

# -----------------------------
# 4) Transaction costs
# -----------------------------
APPLY_T_COSTS = True
COST_PER_TURNOVER = 0.0025

# -----------------------------
# 5) Output
# -----------------------------
RESULTS_DIR = Path("results")
OUTPUT_XLSX = RESULTS_DIR / "utility_results.xlsx"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1) Data
    rets = download_monthly_returns(list(TICKERS.values()), start=START, end=END)
    inv_map = {v: k for k, v in TICKERS.items()}
    rets = rets.rename(columns=inv_map)

    # 2) Clean / coverage
    rets = prepare_returns(rets, min_coverage=MIN_COVERAGE)

    # 3) Rolling utility optimisation
    W, R, meta, diag = rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=1.0,
        rf_col=None,
        optimizer=opt_mean_variance_utility,
        optimizer_kwargs={"risk_aversion": RISK_AVERSION},
    )

    print("--------------------------------------------------")
    print(f"Rolling Mean–Variance Utility | lambda={RISK_AVERSION}")
    print("W shape:", W.shape)
    print("Period:", W.index.min(), "->", W.index.max())
    print("Unique weight rows (6dp):", len(W.round(6).drop_duplicates()))
    print("Avg abs monthly turnover:", float(W.diff().abs().sum(axis=1).mean()))

    if diag is not None and len(diag) > 0 and "success" in diag.columns:
        success_rate = float(diag["success"].fillna(False).mean())
        print("Solver success rate:", round(success_rate, 3))
        print("Fallback share:", round(1 - success_rate, 3))

    print("---- Performance (GROSS) ----")
    print(perf_summary(R))

    extra: dict[str, pd.DataFrame] = {}
    export_R = R

    if APPLY_T_COSTS:
        tc = apply_transaction_costs(W, R, cost_per_turnover=COST_PER_TURNOVER)
        export_R = tc["net_return"].rename("oos_return")
        extra["Transaction_Costs"] = tc
        print("---- Performance (NET) ----")
        print(perf_summary(tc["net_return"]))

    extra["Tail_GROSS"] = tail_metrics(R, alpha=0.95).to_frame("value")
    if APPLY_T_COSTS:
        extra["Tail_NET"] = tail_metrics(export_R, alpha=0.95).to_frame("value")

    if diag is not None and len(diag) > 0:
        extra["Diagnostics"] = diag

    # 4) Export
    export_excel(
        path=str(OUTPUT_XLSX),
        monthly_returns=rets,
        weights=W,
        oos_returns=export_R,
        bounds_meta=meta,
        extra_sheets=extra if extra else None,
    )

    print(f"Saved: {OUTPUT_XLSX.resolve()}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
