# Rolling_markowitz.py
# pip install yfinance pandas numpy scipy openpyxl

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

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
MIN_COVERAGE = 1.0       # 1.0 strict, <1.0 allows some missing

# -----------------------------
# 2) BOUNDS (user parameters)
# -----------------------------
# Set (min,max) weight bounds per asset class.
# Must be feasible: sum(min) <= 1 <= sum(max)
BOUNDS = {
    "SP500":     (0.10, 0.70),
    "UST_7_10Y": (0.00, 0.60),
    "T_BILLS":   (0.00, 0.30),
    "CHINA":     (0.00, 0.30),
    "REITS":     (0.00, 0.30),
    "GOLD":      (0.00, 0.30),
}

# If True, any asset not in BOUNDS gets default bounds below
DEFAULT_BOUNDS = (0.0, 1.0)

# Long-only implied by bounds >= 0; allow shorting by setting negative mins if you want
# e.g. ("SP500": (-0.2, 0.8))

# -----------------------------
# 3) Data: monthly returns
# -----------------------------
def download_monthly_returns(tickers: dict, start: str, end=None) -> pd.DataFrame:
    yf_tickers = list(tickers.values())

    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if "Close" in data:
        px = data["Close"]
    elif "Adj Close" in data:
        px = data["Adj Close"]
    else:
        raise ValueError("Could not find Close/Adj Close in yfinance output.")

    if isinstance(px, pd.Series):
        px = px.to_frame()

    inv_map = {v: k for k, v in tickers.items()}
    px = px.rename(columns=inv_map)

    px_m = px.resample("M").last()
    rets_m = px_m.pct_change().dropna(how="all")
    return rets_m

# -----------------------------
# 4) Helper: bounds handling
# -----------------------------
def build_bounds(columns, bounds_dict, default_bounds=(0.0, 1.0)):
    b = []
    for c in columns:
        b.append(bounds_dict.get(c, default_bounds))
    return b

def validate_bounds(columns, bounds_dict, default_bounds=(0.0, 1.0)):
    mins = []
    maxs = []
    for c in columns:
        lo, hi = bounds_dict.get(c, default_bounds)
        if lo > hi:
            raise ValueError(f"Invalid bounds for {c}: min {lo} > max {hi}")
        mins.append(lo)
        maxs.append(hi)
    if sum(mins) - 1.0 > 1e-9:
        raise ValueError(f"Infeasible: sum(min_bounds)={sum(mins):.4f} > 1.0")
    if 1.0 - sum(maxs) > 1e-9:
        raise ValueError(f"Infeasible: sum(max_bounds)={sum(maxs):.4f} < 1.0")

# -----------------------------
# 5) Markowitz: max Sharpe on EXCESS returns (rf = T_BILLS)
# -----------------------------
def max_sharpe_excess_weights(
    returns: pd.DataFrame,
    rf: pd.Series,
    bounds: list[tuple[float, float]],
) -> pd.Series:
    """
    Maximizes Sharpe ratio using excess returns (returns - rf).
    Bounds are per-asset (min,max) and passed as a list aligned to returns.columns.
    """
    cols = list(returns.columns)

    rf = rf.reindex(returns.index).fillna(0.0)
    excess = returns.sub(rf, axis=0)

    mu = excess.mean().values
    cov = excess.cov().values
    n = len(mu)

    cov = cov + np.eye(n) * 1e-10

    def neg_sharpe(w):
        port_mu = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol <= 0 or not np.isfinite(port_vol):
            return 1e9
        return -(port_mu / port_vol)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Initial guess: project equal weights into bounds roughly
    w0 = np.repeat(1.0 / n, n)
    # simple clip + renormalize
    w0 = np.clip(w0, [lo for lo, _ in bounds], [hi for _, hi in bounds])
    if w0.sum() == 0:
        w0 = np.repeat(1.0 / n, n)
    else:
        w0 = w0 / w0.sum()

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if (not res.success) or np.any(~np.isfinite(res.x)):
        w = pd.Series(w0, index=cols)
        return w / w.sum()

    w = pd.Series(res.x, index=cols)
    w[np.abs(w) < 1e-8] = 0.0

    if w.sum() == 0:
        w = pd.Series(w0, index=cols)
    else:
        w = w / w.sum()

    return w

# -----------------------------
# 6) Rolling backtest
# -----------------------------
def rolling_markowitz_backtest(
    rets: pd.DataFrame,
    lookback_months: int = 120,
    min_coverage: float = 1.0,
    bounds_dict: dict[str, tuple[float, float]] | None = None,
    default_bounds: tuple[float, float] = (0.0, 1.0),
):
    rets = rets.copy()

    if min_coverage < 1.0:
        min_non_nan = int(np.ceil(min_coverage * rets.shape[1]))
        rets = rets.dropna(thresh=min_non_nan)
    else:
        rets = rets.dropna(how="any")

    rets = rets.dropna(axis=1, how="all")

    if "T_BILLS" not in rets.columns:
        raise ValueError("T_BILLS column missing. Ensure TICKERS includes T_BILLS and data downloaded correctly.")

    cols = list(rets.columns)

    if bounds_dict is None:
        bounds_dict = {}

    # Validate feasibility given the actual columns present
    validate_bounds(cols, bounds_dict, default_bounds)
    bounds_list = build_bounds(cols, bounds_dict, default_bounds)

    dates = rets.index
    if len(dates) <= lookback_months + 1:
        raise ValueError(
            f"Not enough monthly data after filtering. "
            f"Need > {lookback_months+1} rows, got {len(dates)}. "
            f"Try earlier START date or set MIN_COVERAGE < 1.0."
        )

    weights_rows = []
    oos_list = []

    for i in range(lookback_months, len(dates) - 1):
        train = rets.iloc[i - lookback_months:i]
        test_date = dates[i + 1]
        test_ret = rets.iloc[i + 1]

        rf_train = train["T_BILLS"]

        w = max_sharpe_excess_weights(train, rf=rf_train, bounds=bounds_list)
        port_ret = float((w * test_ret).sum())

        weights_rows.append(pd.DataFrame([w.values], index=[test_date], columns=w.index))
        oos_list.append(pd.Series([port_ret], index=[test_date]))

    W = pd.concat(weights_rows).sort_index()
    R = pd.concat(oos_list).sort_index()
    R.name = "portfolio_return"

    return W, R, bounds_list

def perf_summary(port_rets: pd.Series) -> pd.Series:
    m = port_rets.dropna()

    ann_ret = (1 + m).prod() ** (12 / len(m)) - 1
    ann_vol = m.std() * np.sqrt(12)
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol

    equity = (1 + m).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())

    return pd.Series({
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_rf0": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": max_dd,
        "months": int(len(m)),
    })

# -----------------------------
# 7) Run + Excel export
# -----------------------------
if __name__ == "__main__":
    rets = download_monthly_returns(TICKERS, START, END)
    print("Monthly returns sample:")
    print(rets.tail())

    W, R, bounds_list = rolling_markowitz_backtest(
        rets,
        lookback_months=LOOKBACK_MONTHS,
        min_coverage=MIN_COVERAGE,
        bounds_dict=BOUNDS,
        default_bounds=DEFAULT_BOUNDS
    )

    print("\nPerformance summary (OOS):")
    print(perf_summary(R))

    equity = (1 + R).cumprod().rename("equity_curve")
    out = pd.concat([R, equity], axis=1)

    # Make bounds sheet
    bounds_df = pd.DataFrame(
        {
            "asset": list(rets.columns),
            "min_weight": [b[0] for b in bounds_list],
            "max_weight": [b[1] for b in bounds_list],
        }
    )

    excel_path = "rolling_markowitz_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="OOS_Returns_Equity", index=True)
        W.to_excel(writer, sheet_name="Weights", index=True)
        rets.to_excel(writer, sheet_name="Monthly_Returns", index=True)
        bounds_df.to_excel(writer, sheet_name="Bounds", index=False)

    print(f"\nSaved: {excel_path}")
