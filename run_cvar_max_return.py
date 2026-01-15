# run_cvar_max_return.py
# Rolling: maximize expected return subject to CVaR(alpha) <= MONTHLY cap
#
# pip install -r requirements.txt
# or: pip install yfinance pandas numpy scipy openpyxl

from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.data import download_monthly_returns, prepare_returns
from src.optimizers import opt_cvar_cap_max_return_monthly
from src.backtest import rolling_optimize, apply_transaction_costs
from src.reporting import perf_summary, tail_metrics, export_excel


# =========================================================
# Settings
# =========================================================
TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "BIL",  # investable cash proxy in THIS runner (rf_col=None)
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GLD",
}

START = "2005-01-01"
END = None

LOOKBACK_MONTHS = 60
MIN_COVERAGE = 1.0  # strict

BOUNDS = {
    "SP500": (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "T_BILLS": (0.00, 0.30),
    "CHINA": (0.00, 0.30),
    "REITS": (0.00, 0.30),
    "GOLD": (0.00, 0.50),
}
DEFAULT_BOUNDS = (0.0, 1.0)

# ---- CVaR parameters ----
ALPHA = 0.90

CVAR_CAP_MONTHLY = 0.0457  # e.g. calibrated to 60/40 CVaR_90

# ---- Transaction costs ----
# 0.0025 = 25 bps cost for 100% monthly turnover
APPLY_T_COSTS = True
COST_PER_TURNOVER = 0.0025

# ---- Output ----
RESULTS_DIR = Path("results")
OUTPUT_XLSX = RESULTS_DIR / "cvar_max_return_results.xlsx"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1) Data (monthly returns)
    rets = download_monthly_returns(list(TICKERS.values()), start=START, end=END)

    # Map yahoo tickers -> your names
    inv_map = {v: k for k, v in TICKERS.items()}
    rets = rets.rename(columns=inv_map)

    # Clean / coverage
    rets = prepare_returns(rets, min_coverage=MIN_COVERAGE)

    # 2) Rolling CVaR-cap optimization (maximize return)
    W, R, meta, diag = rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=1.0,  # already cleaned above
        rf_col=None,
        optimizer=opt_cvar_cap_max_return_monthly,
        optimizer_kwargs={
            "alpha": ALPHA,
            "cvar_cap_monthly": CVAR_CAP_MONTHLY,
        },
    )

    # 3) Transaction costs
    if APPLY_T_COSTS:
        tc = apply_transaction_costs(W, R, cost_per_turnover=COST_PER_TURNOVER)
        R_gross = tc["gross_return"].rename("portfolio_return_gross")
        R_net = tc["net_return"].rename("portfolio_return_net")
    else:
        tc = None
        R_gross = R.rename("portfolio_return_gross")
        R_net = None

    # 4) Console diagnostics
    print("--------------------------------------------------")
    print(f"CVaR-cap max return | alpha={ALPHA:.0%} | monthly cap={CVAR_CAP_MONTHLY:.2%}")
    print("W shape:", W.shape)
    print("Period:", W.index.min(), "->", W.index.max())
    print("Unique weight rows (6dp):", len(W.round(6).drop_duplicates()))
    print("Avg abs monthly turnover:", float(W.diff().abs().sum(axis=1).mean()))
    if APPLY_T_COSTS:
        print(f"Cost per 1.0 turnover: {COST_PER_TURNOVER:.4f}")

    if diag is not None and len(diag) > 0 and "success" in diag.columns:
        success_rate = float(diag["success"].fillna(False).mean())
        print("Solver success rate:", round(success_rate, 3))
        print("Fallback share:", round(1 - success_rate, 3))

    print("---- Performance (GROSS) ----")
    print(perf_summary(R_gross))

    if APPLY_T_COSTS and R_net is not None:
        print("---- Performance (NET) ----")
        print(perf_summary(R_net))

    print("---- Tail metrics (GROSS) ----")
    tm_gross = tail_metrics(R_gross, alpha=ALPHA)
    print(tm_gross)

    if APPLY_T_COSTS and R_net is not None:
        print("---- Tail metrics (NET) ----")
        tm_net = tail_metrics(R_net, alpha=ALPHA)
        print(tm_net)

    print("--------------------------------------------------")

    # 5) Export to Excel
    extra: dict[str, pd.DataFrame] = {}

    if diag is not None and len(diag) > 0:
        extra["Diagnostics"] = diag

    if tc is not None:
        extra["Transaction_Costs"] = tc
        extra["OOS_Returns_NET"] = pd.DataFrame({"net_return": R_net})

    extra["Tail_GROSS"] = tm_gross.to_frame("value")
    if APPLY_T_COSTS and R_net is not None:
        extra["Tail_NET"] = tm_net.to_frame("value")

    export_excel(
        path=str(OUTPUT_XLSX),
        monthly_returns=rets,
        weights=W,
        oos_returns=R_gross.rename("oos_return"),  # main sheet = gross for consistency
        bounds_meta=meta,
        extra_sheets=extra,
    )

    print(f"Saved: {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
