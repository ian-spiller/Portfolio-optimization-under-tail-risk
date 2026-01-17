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

from src.factors import MACRO_FACTORS_V1, build_factor_returns
from src.factor_model import ols_regression, rolling_ols

# NEW: scenario simulation + regimes
from src.scenarios import (
    run_factor_scenarios,
    SimulationSpec,
    TailSpec,
    ScenarioSpec,
    RegimeSpec,
)

# -----------------------------
# 1) Universe (Yahoo tickers)
# -----------------------------
TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
    "T_BILLS": "SHY",     # used as rf proxy
    "CHINA": "FXI",
    "REITS": "VNQ",
    "GOLD": "GC=F",
}

START = "2005-01-01"
END = None
LOOKBACK_MONTHS = 60
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

FACTOR_WINDOW = 60


def main():
    # 1) Download monthly returns for all tickers
    rets = download_monthly_returns(list(TICKERS.values()), start=START, end=END)

    # Map yahoo tickers -> your names
    inv_map = {v: k for k, v in TICKERS.items()}
    rets = rets.rename(columns=inv_map)

    # 2) Clean / coverage
    rets = prepare_returns(rets, min_coverage=MIN_COVERAGE, keep_cols=["T_BILLS"])

    if "T_BILLS" not in rets.columns:
        raise ValueError("T_BILLS missing from returns. Check ticker mapping and data availability.")

    # 3) Rolling optimization using rf as separate input
    W, R, meta, diag = rolling_optimize(
        rets=rets,
        lookback_months=LOOKBACK_MONTHS,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS,
        min_coverage=1.0,
        rf_col="T_BILLS",
        optimizer=opt_max_sharpe_excess,
        optimizer_kwargs={},
    )

    print("\nPerformance summary (gross):")
    print(perf_summary(R))

    extra_sheets = {"Diagnostics": diag} if (diag is not None and not diag.empty) else {}

    # 4) Optional transaction costs
    if APPLY_T_COSTS:
        net = apply_transaction_costs(W, R, cost_per_turnover=COST_PER_TURNOVER)
        print("\nPerformance summary (net):")
        print(perf_summary(net["net_return"]))
        extra_sheets["Net_Returns"] = net
        export_R = net["net_return"].rename("oos_return")
    else:
        export_R = R

    if isinstance(export_R, pd.DataFrame):
        export_R = export_R.squeeze("columns")
    export_R = export_R.rename("oos_return").astype(float)

    try:
        f_start = str(export_R.index.min().date())
        f_end = str(export_R.index.max().date())

        factors = build_factor_returns(MACRO_FACTORS_V1, start=f_start, end=f_end)

        # Align to portfolio returns (avoid factor NA causing short beta history)
        factors = factors.reindex(export_R.index).dropna(how="any")
        export_R_aligned = export_R.reindex(factors.index)

        reg = ols_regression(export_R_aligned, factors, add_const=True)
        roll = rolling_ols(export_R_aligned, factors, window=FACTOR_WINDOW, add_const=True)

        extra_sheets["Macro_Factors"] = factors
        extra_sheets["Factor_Regression_Coef"] = reg["coef"]
        extra_sheets["Factor_Regression_Fit"] = reg["fit"]
        extra_sheets["Rolling_Betas"] = roll

        # --------------------------------------------------
        # Scenario simulation (factor-based) + regime covariance
        # --------------------------------------------------
        rep = run_factor_scenarios(
            factor_returns=factors,
            rolling_betas=roll,
            asof=export_R_aligned.index.max(),  # avoid look-ahead
            beta_pick="asof",
            sim=SimulationSpec(
                method="student_t",
                df=6,
                n_sims=20_000,
                lookback_months=60,
                horizon_months=1,
                seed=42,
                demean=True,
            ),
            regime=RegimeSpec(
                method="rolling_vol",
                factor_for_vol="RISK_ON",
                vol_window=12,
                threshold_quantile=0.70,
                regime="current",          # "current" / "high" / "low"
                min_obs_per_regime=24,
            ),
            tail=TailSpec(alpha=0.95, loss_thresholds=(0.02, 0.05)),
            stress_scenarios=[
                ScenarioSpec("Risk-off", {"RISK_ON": -0.08, "CREDIT": -0.03, "USD": +0.02, "RATES": +0.02}),
                ScenarioSpec("Inflation shock", {"INFLATION": +0.03, "RATES": -0.02, "RISK_ON": -0.04}),
                ScenarioSpec("Credit widening", {"CREDIT": -0.04, "RISK_ON": -0.03}),
            ],
        )

        extra_sheets["Scenario_Summary"] = rep.summary
        extra_sheets["Scenario_Tail_Decomp"] = rep.tail_decomp
        if rep.stress_table is not None:
            extra_sheets["Scenario_Stress_Table"] = rep.stress_table

        extra_sheets["Scenario_Sim_Returns_Sample"] = rep.sim_returns.sample(
            n=min(5000, len(rep.sim_returns)),
            random_state=42,
        ).to_frame("sim_return")

    except Exception as e:
        extra_sheets["Factor_Error"] = pd.DataFrame({"error": [str(e)]})

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
