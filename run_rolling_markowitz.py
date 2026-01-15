# run_rolling_markowitz.py
# pip install yfinance pandas numpy scipy openpyxl

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.data import download_monthly_returns, prepare_returns
from src.optimizers import opt_max_sharpe_excess
from src.backtest import rolling_optimize, apply_transaction_costs
from src.reporting import perf_summary, export_excel

# -----------------------------
# 1) Universe (Yahoo tickers)
# -----------------------------
TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "BIL",     # used as rf proxy
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GC=F",
}

START = "2005-01-01"
END = None
LOOKBACK_MONTHS = 120
MIN_COVERAGE = 1.0  # 1.0 strict

# -----------------------------
# 2) BOUNDS
# -----------------------------
BOUNDS = {
    "SP500":     (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "CHINA":     (0.00, 0.30),
    "REITS":     (0.00, 0.30),
    "GOLD":      (0.00, 0.30),
    # Note: do NOT set bounds for T_BILLS here because it is used as rf, not optimized as an asset.
}

DEFAULT_BOUNDS = (0.0, 1.0)

# Optional: transaction cost model
APPLY_T_COSTS = True
COST_PER_TURNOVER = 0.0025  # 25 bps per 1.0 turnover


def main():
    # 1) Download monthly returns for all tickers
    rets = download_monthly_returns(list(TICKERS.values()), start=START, end=END)
    # Map yahoo tickers -> your names
    inv_map = {v: k for k, v in TICKERS.items()}
    rets = rets.rename(columns=inv_map)

    # 2) Clean / coverage
    # Keep all columns for now; prepare_returns will drop assets if MIN_COVERAGE < 1.0
    rets = prepare_returns(rets, min_coverage=MIN_COVERAGE, keep_cols=["T_BILLS"])

    if "T_BILLS" not in rets.columns:
        raise ValueError("T_BILLS missing from returns. Check ticker mapping and data availability.")

    # 3) Rolling optimization using rf as separate input
    W, R, meta, diag = rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=1.0,           # already cleaned above
        rf_col="T_BILLS",
        optimizer=opt_max_sharpe_excess,
        optimizer_kwargs={},        # none needed
    )

    print("\nPerformance summary (gross):")
    print(perf_summary(R))

    extra_sheets = {"Diagnostics": diag} if not diag.empty else {}

    # 4) Optional transaction costs
    if APPLY_T_COSTS:
        net = apply_transaction_costs(W, R, cost_per_turnover=COST_PER_TURNOVER)
        print("\nPerformance summary (net):")
        print(perf_summary(net["net_return"]))
        extra_sheets["Net_Returns"] = net
        # Export net series as “official” return series in the workbook
        export_R = net["net_return"].rename("oos_return")
    else:
        export_R = R

    # 5) Export

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    excel_path = results_dir / "rolling_markowitz_results.xlsx"

    export_excel(
        path=excel_path,
        monthly_returns=rets,
        weights=W,
        oos_returns=export_R,
        bounds_meta=meta,
        extra_sheets=extra_sheets,
    )

    print(f"\nSaved: {excel_path.resolve()}")


if __name__ == "__main__":
    main()