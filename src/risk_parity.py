# run_risk_parity.py
# Runs rolling Risk Parity (Equal Risk Contribution) using core.py
#
# pip install yfinance pandas numpy scipy openpyxl

import core

TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "BIL",
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GC=F",
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

OUTPUT_XLSX = "results_risk_parity.xlsx"

if __name__ == "__main__":
    rets = core.download_monthly_returns(TICKERS, START, END)

    # NOTE: core.rolling_optimize now returns (W, R, meta, diag)
    W, R, meta, diag = core.rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=MIN_COVERAGE,
        rf_col=None,
        optimizer=core.opt_risk_parity,
        optimizer_kwargs={},  # none
    )

    # Quick diagnostics in console
    print("W shape:", W.shape)
    print("W date range:", W.index.min(), "->", W.index.max())
    print("Unique weight rows (rounded 6dp):", len(W.round(6).drop_duplicates()))
    print("Avg abs monthly turnover:", W.diff().abs().sum(axis=1).mean())

    # Export diagnostics sheet too
    extra = {"Diagnostics": diag} if (diag is not None and len(diag) > 0) else None

    core.export_excel(
        path=OUTPUT_XLSX,
        monthly_returns=rets,
        weights=W,
        oos_returns=R,
        bounds_meta=meta,
        extra_sheets=extra,
    )

    print(f"Saved: {OUTPUT_XLSX}")
    print(core.perf_summary(R))
