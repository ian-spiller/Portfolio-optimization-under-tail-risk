# run_cvar_target_risk.py
# Rolling: maximize expected return subject to CVaR(alpha) <= MONTHLY cap
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
    "T_BILLS": "BIL",
    "REITS": "VNQ",
    "GOLD": "GLD",
}

START = "2005-01-01"
END = None

LOOKBACK_MONTHS = 120
MIN_COVERAGE = 1.0

BOUNDS = {
    "SP500":     (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "T_BILLS":   (0.00, 0.30),
    "REITS":     (0.00, 0.30),
    "GOLD":      (0.00, 0.50),
}
DEFAULT_BOUNDS = (0.0, 1.0)

# ---- CVaR parameters ----
ALPHA = 0.9

# Monthly CVaR cap
# Interpretacija: povprečna izguba v najslabših (1-ALPHA)% mesecih
CVAR_CAP_MONTHLY = 0.0457  # npr. kalibrirano na 60/40 CVaR₉₀

# ---- Transaction costs ----
# 0.0025 = 25 bps stroška za 100% mesečni turnover
COST_PER_TURNOVER = 0.0025

OUTPUT_XLSX = "results_cvar_max_ret.xlsx"

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    # 1) Data
    rets = core.download_monthly_returns(TICKERS, START, END)

    # 2) Rolling CVaR-target optimization
    W, R, meta, diag = core.rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=MIN_COVERAGE,
        rf_col=None,
        optimizer=core.opt_cvar_cap_max_return_monthly,
        optimizer_kwargs={
            "alpha": ALPHA,
            "cvar_cap_monthly": CVAR_CAP_MONTHLY,
        },
    )

    # 3) Transaction costs: Gross vs Net
    tc = core.apply_transaction_costs(
        weights=W,
        oos_returns=R,
        cost_per_turnover=COST_PER_TURNOVER,
    )

    R_gross = tc["gross_return"]
    R_net = tc["net_return"]

    R_gross.name = "portfolio_return_gross"
    R_net.name = "portfolio_return_net"

    # =====================================================
    # Console diagnostics
    # =====================================================
    print("--------------------------------------------------")
    print(f"CVaR-target risk | alpha={ALPHA:.0%} | monthly cap={CVAR_CAP_MONTHLY:.2%}")
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

    print("---- Tail metrics (GROSS) ----")
    tm_gross = core.tail_metrics(R_gross, alpha=ALPHA)
    print(tm_gross)

    print("---- Tail metrics (NET) ----")
    tm_net = core.tail_metrics(R_net, alpha=ALPHA)
    print(tm_net)

    print("--------------------------------------------------")

    # =====================================================
    # Export to Excel
    # =====================================================
    extra = {}

    if diag is not None and len(diag) > 0:
        extra["Diagnostics"] = diag

    extra["Transaction_Costs"] = tc
    extra["OOS_Returns_NET"] = pd.DataFrame({"net_return": R_net})
    extra["Tail_GROSS"] = tm_gross.to_frame("value")
    extra["Tail_NET"] = tm_net.to_frame("value")

    core.export_excel(
        path=OUTPUT_XLSX,
        monthly_returns=rets,
        weights=W,
        oos_returns=R_gross,   # osnovni sheet = GROSS (kompatibilno z ostalimi runnerji)
        bounds_meta=meta,
        extra_sheets=extra,
    )

    print(f"Saved: {OUTPUT_XLSX}")
