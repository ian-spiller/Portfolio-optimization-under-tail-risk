# run_sharpe.py
# Rolling Max-Sharpe (excess) optimization + transaction costs
#
# pip install yfinance pandas numpy scipy openpyxl

import pandas as pd
import core

# =========================================================
# Settings
# =========================================================
TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "BIL",   # rf proxy
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GC=F",     # priporočilo: GLD za manj NaN
}

START = "2005-01-01"
END = None

LOOKBACK_MONTHS = 120
MIN_COVERAGE = 1.0

BOUNDS = {
    "SP500":     (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "T_BILLS":   (0.00, 0.30),
    "CHINA":     (0.00, 0.30),
    "REITS":     (0.00, 0.30),
    "GOLD":      (0.00, 0.30),
}
DEFAULT_BOUNDS = (0.0, 1.0)

RF_COL = "T_BILLS"

# ---- NEW: transaction costs ----
COST_PER_TURNOVER = 0.0025   # 25 bps per 100% turnover

OUTPUT_XLSX = "results_sharpe.xlsx"

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    # 1) Data
    rets = core.download_monthly_returns(TICKERS, START, END)

    # 2) Rolling optimization (gross)
    W, R_gross, meta, diag = core.rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=MIN_COVERAGE,
        rf_col=RF_COL,
        optimizer=core.opt_max_sharpe_excess,
        optimizer_kwargs={},
    )

    R_gross.name = "portfolio_return_gross"

    # =====================================================
    # NEW: Apply transaction costs
    # =====================================================
    tc = core.apply_transaction_costs(
        weights=W,
        oos_returns=R_gross,
        cost_per_turnover=COST_PER_TURNOVER,
    )

    R_net = tc["net_return"]
    R_net.name = "portfolio_return_net"

    # =====================================================
    # Diagnostics (console)
    # =====================================================
    print("--------------------------------------------------")
    print("Sharpe (excess) – diagnostics")
    print("W shape:", W.shape)
    print("Period:", W.index.min(), "->", W.index.max())
    print("Unique weight rows (6dp):", len(W.round(6).drop_duplicates()))
    print("Avg abs monthly turnover:", W.diff().abs().sum(axis=1).mean())
    print(f"Cost per 1.0 turnover: {COST_PER_TURNOVER:.4f}")

    if diag is not None and len(diag) > 0:
        success_rate = diag["success"].fillna(False).mean()
        print("Solver success rate:", round(float(success_rate), 3))
        print("Fallback share:", round(float(1 - success_rate), 3))

    print("---- Performance (GROSS) ----")
    print(core.perf_summary(R_gross))
    print("---- Performance (NET) ----")
    print(core.perf_summary(R_net))
    print("--------------------------------------------------")

    # =====================================================
    # Export (include transaction costs + net returns)
    # =====================================================
    extra = {}
    if diag is not None and len(diag) > 0:
        extra["Diagnostics"] = diag
    extra["Transaction_Costs"] = tc
    extra["OOS_Returns_NET"] = pd.DataFrame({"net_return": R_net})

    core.export_excel(
        path=OUTPUT_XLSX,
        monthly_returns=rets,
        weights=W,
        oos_returns=R_gross,   # base sheet = gross (dosledno z drugimi runnerji)
        bounds_meta=meta,
        extra_sheets=extra if len(extra) > 0 else None,
    )

    print(f"Saved: {OUTPUT_XLSX}")
