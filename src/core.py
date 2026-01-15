# core.py
# Shared "engine" for multi-asset rolling optimizations + Excel export
#
# pip install yfinance pandas numpy scipy openpyxl

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, linprog



# =========================================================
# 1) Data
# =========================================================
def download_monthly_returns(tickers: dict[str, str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Downloads prices from Yahoo Finance and returns month-end simple returns.
    Uses auto_adjust=True (total-return-ish for ETFs).
    """
    yf_tickers = list(tickers.values())

    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
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


# =========================================================
# 2) Bounds utilities
# =========================================================
def build_bounds(
    columns: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    default_bounds: tuple[float, float] = (0.0, 1.0),
) -> list[tuple[float, float]]:
    return [bounds_dict.get(c, default_bounds) for c in columns]


def validate_bounds(
    columns: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    default_bounds: tuple[float, float] = (0.0, 1.0),
) -> None:
    mins, maxs = [], []
    for c in columns:
        lo, hi = bounds_dict.get(c, default_bounds)
        if lo > hi:
            raise ValueError(f"Invalid bounds for {c}: min {lo} > max {hi}")
        mins.append(lo)
        maxs.append(hi)

    if sum(mins) - 1.0 > 1e-9:
        raise ValueError(f"Infeasible bounds: sum(min_bounds)={sum(mins):.4f} > 1.0")
    if 1.0 - sum(maxs) > 1e-9:
        raise ValueError(f"Infeasible bounds: sum(max_bounds)={sum(maxs):.4f} < 1.0")


# =========================================================
# 3) Objective helpers (common stats)
# =========================================================
def _cov_stable(returns: pd.DataFrame) -> np.ndarray:
    cov = returns.cov().values
    return cov + np.eye(cov.shape[0]) * 1e-10


def _clip_and_renormalize(w: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    w = np.clip(w, lo, hi)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
        w = np.clip(w, lo, hi)
        s = w.sum()
    return w / s


# =========================================================
# 4) Optimizers (pluggable)
#    Each optimizer returns either:
#      - pd.Series weights
#      - OR (pd.Series weights, info_dict)  for diagnostics
# =========================================================
def opt_max_sharpe_excess(
    returns: pd.DataFrame,
    rf: pd.Series,
    bounds: list[tuple[float, float]],
) -> tuple[pd.Series, dict]:
    """
    Max Sharpe on excess returns: (returns - rf). Sharpe computed with rf=0 on excess.
    Returns (weights, info) so rolling_optimize can log solver diagnostics.
    """
    cols = list(returns.columns)
    rf = rf.reindex(returns.index).fillna(0.0)
    excess = returns.sub(rf, axis=0)

    mu = excess.mean().values
    cov = _cov_stable(excess)
    n = len(mu)

    def neg_sharpe(w):
        port_mu = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol <= 0 or not np.isfinite(port_vol):
            return 1e9
        return -(port_mu / port_vol)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = _clip_and_renormalize(np.ones(n) / n, bounds)

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        w = w0
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "w0"}
    else:
        w = res.x
        info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}

    w = _clip_and_renormalize(w, bounds)
    return pd.Series(w, index=cols), info


def opt_min_variance(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
) -> tuple[pd.Series, dict]:
    """
    Minimum variance portfolio.
    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)
    cov = _cov_stable(returns)
    n = cov.shape[0]

    def var(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = _clip_and_renormalize(np.ones(n) / n, bounds)

    res = minimize(var, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        w = w0
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "w0"}
    else:
        w = res.x
        info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}

    w = _clip_and_renormalize(w, bounds)
    return pd.Series(w, index=cols), info


def opt_mean_variance_utility(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    risk_aversion: float = 5.0,
) -> tuple[pd.Series, dict]:
    """
    Maximize: mu'w - (lambda/2) w'Sigma w
    risk_aversion = lambda (higher => more conservative).
    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)
    mu = returns.mean().values
    cov = _cov_stable(returns)
    n = len(mu)
    lam = float(risk_aversion)

    def neg_utility(w):
        util = float(mu @ w) - 0.5 * lam * float(w @ cov @ w)
        return -util

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = _clip_and_renormalize(np.ones(n) / n, bounds)

    res = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        w = w0
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "w0"}
    else:
        w = res.x
        info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}

    w = _clip_and_renormalize(w, bounds)
    return pd.Series(w, index=cols), info


def opt_target_vol(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    target_vol_annual: float = 0.10,
    vol_constraint: str = "ineq",  # "ineq" => vol <= target, "eq" => vol = target
) -> tuple[pd.Series, dict]:
    """
    Target-vol portfolio:
      maximize expected return subject to portfolio volatility constraint.

    vol_constraint:
      - "ineq": sqrt(w'Σw) <= target_vol_monthly
      - "eq":   sqrt(w'Σw) =  target_vol_monthly (can be infeasible with bounds)

    If infeasible or solver fails -> fallback to min-variance.
    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)
    mu = returns.mean().values
    cov = _cov_stable(returns)
    n = len(mu)

    target_vol_m = float(target_vol_annual) / np.sqrt(12.0)

    def port_vol(w):
        v = float(np.sqrt(w @ cov @ w))
        if not np.isfinite(v):
            return 1e9
        return v

    def neg_return(w):
        return -float(mu @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if vol_constraint == "eq":
        cons.append({"type": "eq", "fun": lambda w: target_vol_m - port_vol(w)})
    else:
        cons.append({"type": "ineq", "fun": lambda w: target_vol_m - port_vol(w)})

    w0 = _clip_and_renormalize(np.ones(n) / n, bounds)
    res = minimize(neg_return, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(returns, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = _clip_and_renormalize(res.x, bounds)
    info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}
    return pd.Series(w, index=cols), info


def opt_risk_parity(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
) -> tuple[pd.Series, dict]:
    """
    Equal Risk Contribution (ERC) / Risk Parity:
      minimize sum_i (RC_i - RC_bar)^2
    with RC_i = w_i * (Sigma w)_i and RC_bar = portfolio_var / n

    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)
    cov = _cov_stable(returns)
    n = cov.shape[0]

    def objective(w):
        w = np.asarray(w, dtype=float)
        m = cov @ w
        port_var = float(w @ m)
        if not np.isfinite(port_var) or port_var <= 0:
            return 1e9
        rc = w * m
        rc_bar = port_var / n
        return float(np.sum((rc - rc_bar) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = _clip_and_renormalize(np.ones(n) / n, bounds)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(returns, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = _clip_and_renormalize(res.x, bounds)
    info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}
    return pd.Series(w, index=cols), info


# =========================================================
# 5) Rolling engine
# =========================================================
def prepare_returns(
    rets: pd.DataFrame,
    min_coverage: float = 1.0,
) -> pd.DataFrame:
    """
    Cleans returns by dropping months with too many NaNs.
    """
    rets = rets.copy()

    if min_coverage < 1.0:
        min_non_nan = int(np.ceil(min_coverage * rets.shape[1]))
        rets = rets.dropna(thresh=min_non_nan)
    else:
        rets = rets.dropna(how="any")

    rets = rets.dropna(axis=1, how="all")
    return rets


def rolling_optimize(
    rets: pd.DataFrame,
    lookback_months: int,
    bounds_dict: dict[str, tuple[float, float]],
    default_bounds: tuple[float, float] = (0.0, 1.0),
    min_coverage: float = 1.0,
    rf_col: str | None = None,
    optimizer=None,
    optimizer_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Rolling optimization engine.

    Returns:
      W: weights (indexed by OOS month)
      R: OOS portfolio returns (Series)
      meta: bounds used
      diag: per-month solver diagnostics (success/fallback/message + basic sizes)
    """
    if optimizer is None:
        raise ValueError("optimizer must be provided")

    optimizer_kwargs = optimizer_kwargs or {}

    rets = prepare_returns(rets, min_coverage=min_coverage)
    cols = list(rets.columns)

    validate_bounds(cols, bounds_dict, default_bounds)
    bounds_list = build_bounds(cols, bounds_dict, default_bounds)

    if rf_col is not None and rf_col not in cols:
        raise ValueError(f"rf_col '{rf_col}' not found in returns columns.")

    dates = rets.index
    if len(dates) <= lookback_months + 1:
        raise ValueError(
            f"Not enough monthly data after filtering. Need > {lookback_months+1}, got {len(dates)}."
        )

    weights_rows = []
    oos_list = []
    diag_rows = []

    for i in range(lookback_months, len(dates) - 1):
        train = rets.iloc[i - lookback_months:i]
        test_date = dates[i + 1]
        test_ret = rets.iloc[i + 1]

        if rf_col is None:
            res_opt = optimizer(train, bounds_list, **optimizer_kwargs)
        else:
            rf_train = train[rf_col]
            res_opt = optimizer(train, rf_train, bounds_list, **optimizer_kwargs)

        if isinstance(res_opt, tuple) and len(res_opt) == 2:
            w, info = res_opt
        else:
            w, info = res_opt, {}

        # Ensure alignment
        w = w.reindex(cols).fillna(0.0)
        if float(w.sum()) == 0:
            w = pd.Series(np.ones(len(cols)) / len(cols), index=cols)
        else:
            w = w / w.sum()

        port_ret = float((w * test_ret).sum())

        weights_rows.append(pd.DataFrame([w.values], index=[test_date], columns=cols))
        oos_list.append(pd.Series([port_ret], index=[test_date]))

        diag_rows.append({
            "date": test_date,
            "success": info.get("success", None),
            "fallback": info.get("fallback", ""),
            "message": info.get("message", ""),
            "train_rows": int(train.shape[0]),
            "train_cols": int(train.shape[1]),
        })

    W = pd.concat(weights_rows).sort_index()
    R = pd.concat(oos_list).sort_index()
    R.name = "portfolio_return"

    meta = pd.DataFrame({
        "asset": cols,
        "min_weight": [b[0] for b in bounds_list],
        "max_weight": [b[1] for b in bounds_list],
    })

    diag = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()

    return W, R, meta, diag


# =========================================================
# 6) Reporting / Excel
# =========================================================
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


def export_excel(
    path: str,
    monthly_returns: pd.DataFrame,
    weights: pd.DataFrame,
    oos_returns: pd.Series,
    bounds_meta: pd.DataFrame,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> None:
    equity = (1 + oos_returns).cumprod().rename("equity_curve")
    out = pd.concat([oos_returns, equity], axis=1)

    summary_df = perf_summary(oos_returns).to_frame("value")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="OOS_Returns_Equity", index=True)
        weights.to_excel(writer, sheet_name="Weights", index=True)
        monthly_returns.to_excel(writer, sheet_name="Monthly_Returns", index=True)
        bounds_meta.to_excel(writer, sheet_name="Bounds", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=True)

        if extra_sheets:
            for name, df in extra_sheets.items():
                df.to_excel(writer, sheet_name=name, index=True)

# =========================================================
def opt_cvar_min(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    alpha: float = 0.95,
) -> tuple[pd.Series, dict]:
    """
    Minimize CVaR (Expected Shortfall) at confidence alpha using Rockafellar-Uryasev LP.

    We treat losses L_t = -(r_t' w). Then:
      min_{w,z,u}  z + (1/((1-alpha)T)) * sum(u_t)
      s.t. u_t >= L_t - z
           u_t >= 0
           sum(w)=1
           bounds on w

    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)
    R = returns.dropna(how="any").values  # T x N
    T, N = R.shape

    if T < 5:
        # Too little data: fallback
        w_fallback, _ = opt_min_variance(returns, bounds)
        return w_fallback, {"success": False, "message": "Too few rows for CVaR", "fallback": "min_variance"}

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    c = 1.0 / ((1.0 - alpha) * T)

    # Decision variables: [w_1..w_N, z, u_1..u_T]  => N + 1 + T vars
    nvars = N + 1 + T

    # Objective: minimize z + c * sum(u_t)
    obj = np.zeros(nvars)
    obj[N] = 1.0
    obj[N + 1:] = c

    # Inequalities: u_t >= -(r_t'w) - z  <=>  -u_t - r_t'w - z <= 0
    # Build A_ub x <= b_ub
    A_ub = np.zeros((T, nvars))
    b_ub = np.zeros(T)

    # coefficients for w: -r_t
    A_ub[:, 0:N] = -R
    # coefficient for z: -1
    A_ub[:, N] = -1.0
    # coefficient for u_t: -1 on its own column
    A_ub[np.arange(T), N + 1 + np.arange(T)] = -1.0

    # Equalities: sum(w)=1
    A_eq = np.zeros((1, nvars))
    A_eq[0, 0:N] = 1.0
    b_eq = np.array([1.0])

    # Bounds for variables:
    # w_i in given bounds, z unbounded, u_t >= 0
    var_bounds: list[tuple[float | None, float | None]] = []
    var_bounds.extend([(float(lo), float(hi)) for (lo, hi) in bounds])
    var_bounds.append((None, None))          # z
    var_bounds.extend([(0.0, None)] * T)     # u_t

    res = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=var_bounds,
        method="highs",
    )

    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(returns, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = res.x[:N]
    w = _clip_and_renormalize(w, bounds)

    info = {
        "success": True,
        "message": str(getattr(res, "message", "")),
        "fallback": "",
        "alpha": alpha,
        "T": int(T),
    }
    return pd.Series(w, index=cols), info

# =========================================================
# --- PATCH for core.py: CVaR-target return optimizer (min CVaR s.t. E[r_p] >= target) ---
# 1) Make sure you have this import at the top of core.py:
# from scipy.optimize import minimize, linprog

def opt_cvar_target_return(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    alpha: float = 0.95,
    target_return_annual: float = 0.05,  # e.g. 5% p.a.
) -> tuple[pd.Series, dict]:
    """
    Minimize CVaR (Expected Shortfall) at confidence alpha
    SUBJECT TO a minimum expected return constraint.

    Formulation (Rockafellar-Uryasev LP) with losses L_t = -(r_t' w):
      min_{w,z,u}  z + c * sum(u_t)
      s.t.         -u_t - r_t'w - z <= 0     for all t
                   sum(w)=1
                   mu'w >= target_return_monthly
                   w in bounds, u_t>=0, z free

    Returns (weights, info) for diagnostics.
    """
    cols = list(returns.columns)

    # Use rows with complete data
    df = returns.dropna(how="any")
    R = df.values  # T x N
    T, N = R.shape

    if T < 10:
        w_fallback, _ = opt_min_variance(returns, bounds)
        return w_fallback, {"success": False, "message": "Too few rows for CVaR-target", "fallback": "min_variance"}

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    # Expected returns (monthly) from in-sample window
    mu = df.mean().values  # N

    # Convert annual target to monthly (geometric is overkill here; simple approx)
    target_m = float(target_return_annual) / 12.0

    c = 1.0 / ((1.0 - alpha) * T)

    # Decision variables: [w_1..w_N, z, u_1..u_T]  => N + 1 + T vars
    nvars = N + 1 + T

    # Objective: minimize z + c * sum(u_t)
    obj = np.zeros(nvars)
    obj[N] = 1.0
    obj[N + 1:] = c

    # Inequalities (A_ub x <= b_ub)

    # 1) CVaR constraints: -u_t - r_t'w - z <= 0
    A_ub = np.zeros((T + 1, nvars))
    b_ub = np.zeros(T + 1)

    A_ub[:T, 0:N] = -R
    A_ub[:T, N] = -1.0
    A_ub[np.arange(T), N + 1 + np.arange(T)] = -1.0
    b_ub[:T] = 0.0

    # 2) Expected return constraint: mu'w >= target_m
    # Convert to <= form: -mu'w <= -target_m
    A_ub[T, 0:N] = -mu
    b_ub[T] = -target_m

    # Equalities: sum(w)=1
    A_eq = np.zeros((1, nvars))
    A_eq[0, 0:N] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w in bounds, z free, u >= 0
    var_bounds: list[tuple[float | None, float | None]] = []
    var_bounds.extend([(float(lo), float(hi)) for (lo, hi) in bounds])
    var_bounds.append((None, None))        # z
    var_bounds.extend([(0.0, None)] * T)   # u_t

    res = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=var_bounds,
        method="highs",
    )

    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(returns, bounds)
        info = {
            "success": False,
            "message": str(getattr(res, "message", "")),
            "fallback": "min_variance",
            "alpha": alpha,
            "target_return_annual": target_return_annual,
        }
        return w_fallback, info

    w = res.x[:N]
    w = _clip_and_renormalize(w, bounds)

    info = {
        "success": True,
        "message": str(getattr(res, "message", "")),
        "fallback": "",
        "alpha": alpha,
        "target_return_annual": target_return_annual,
        "target_return_monthly": target_m,
        "T": int(T),
    }
    return pd.Series(w, index=cols), info


# =========================================================

def opt_cvar_cap_max_return(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    alpha: float = 0.95,
    cvar_cap_annual: float = 0.12,   # cap on annualized CVaR (loss), e.g. 12%
) -> tuple[pd.Series, dict]:
    """
    Maximize expected return subject to CVaR(alpha) <= cap.

    Losses: L_t = -(r_t' w)
    Rockafellar-Uryasev:
      CVaR = z + (1/((1-alpha)T)) * sum(u_t)
      u_t >= L_t - z
      u_t >= 0

    We solve LP:
      max mu'w
      s.t. -u_t - r_t'w - z <= 0        for all t
           sum(w)=1
           z + c*sum(u_t) <= cap
           w in bounds, u_t>=0, z free

    Note on scaling:
      CVaR here is on MONTHLY losses. If you pass an ANNUAL cap, we convert to monthly
      using /sqrt(12) (same rough scaling as vol / ES under iid-ish assumptions).
      If you prefer a monthly cap directly, set cvar_cap_annual = monthly_cap * sqrt(12).
    """
    cols = list(returns.columns)

    df = returns.dropna(how="any")
    R = df.values  # T x N
    T, N = R.shape

    if T < 10:
        w_fallback, _ = opt_min_variance(returns, bounds)
        return w_fallback, {"success": False, "message": "Too few rows for CVaR-cap", "fallback": "min_variance"}

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    mu = df.mean().values  # monthly expected returns
    c = 1.0 / ((1.0 - alpha) * T)

    # Convert annual CVaR cap to monthly cap (approx)
    cap_m = float(cvar_cap_annual) / np.sqrt(12.0)

    # Variables: [w_1..w_N, z, u_1..u_T] => N + 1 + T
    nvars = N + 1 + T

    # Objective for linprog (minimize): -mu'w
    obj = np.zeros(nvars)
    obj[:N] = -mu
    # z and u have 0 objective weight

    # Inequalities A_ub x <= b_ub
    # (1) u_t >= -(r_t'w) - z  <=>  -u_t - r_t'w - z <= 0
    # (2) CVaR constraint: z + c*sum(u_t) <= cap_m
    A_ub = np.zeros((T + 1, nvars))
    b_ub = np.zeros(T + 1)

    # (1)
    A_ub[:T, 0:N] = -R
    A_ub[:T, N] = -1.0
    A_ub[np.arange(T), N + 1 + np.arange(T)] = -1.0
    b_ub[:T] = 0.0

    # (2)
    A_ub[T, N] = 1.0
    A_ub[T, N + 1:] = c
    b_ub[T] = cap_m

    # Equalities: sum(w) = 1
    A_eq = np.zeros((1, nvars))
    A_eq[0, 0:N] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w in bounds, z free, u_t >= 0
    var_bounds: list[tuple[float | None, float | None]] = []
    var_bounds.extend([(float(lo), float(hi)) for (lo, hi) in bounds])
    var_bounds.append((None, None))        # z
    var_bounds.extend([(0.0, None)] * T)   # u_t

    res = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=var_bounds,
        method="highs",
    )

    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(returns, bounds)
        info = {
            "success": False,
            "message": str(getattr(res, "message", "")),
            "fallback": "min_variance",
            "alpha": alpha,
            "cvar_cap_annual": float(cvar_cap_annual),
            "cvar_cap_monthly_used": float(cap_m),
        }
        return w_fallback, info

    w = res.x[:N]
    w = _clip_and_renormalize(w, bounds)

    info = {
        "success": True,
        "message": str(getattr(res, "message", "")),
        "fallback": "",
        "alpha": alpha,
        "cvar_cap_annual": float(cvar_cap_annual),
        "cvar_cap_monthly_used": float(cap_m),
        "T": int(T),
    }
    return pd.Series(w, index=cols), info

# =========================================================

def apply_transaction_costs(
    weights: pd.DataFrame,
    oos_returns: pd.Series,
    cost_per_turnover: float = 0.0025,  # 25 bps per 1.0 turnover
) -> pd.DataFrame:
    """
    Computes gross vs net returns given portfolio weights and transaction costs.
    """
    W = weights.copy()
    Rg = oos_returns.copy()

    # turnover_t = sum |w_t - w_{t-1}|
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)

    costs = cost_per_turnover * turnover
    Rn = Rg - costs

    out = pd.DataFrame({
        "gross_return": Rg,
        "net_return": Rn,
        "turnover": turnover,
        "tcost": costs,
    })

    return out



# =========================================================
def opt_cvar_cap_max_return_monthly(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    alpha: float = 0.95,
    cvar_cap_monthly: float = 0.04,   # npr. 4% povp. izguba v worst (1-alpha)% mesecih
) -> tuple[pd.Series, dict]:
    """
    Maximize expected return subject to MONTHLY CVaR(alpha) <= cap (on losses).

    Losses: L_t = -(r_t' w)
    CVaR = z + (1/((1-alpha)T)) * sum(u_t)

    LP:
      max mu'w
      s.t. -u_t - r_t'w - z <= 0        for all t
           sum(w)=1
           z + c*sum(u_t) <= cvar_cap_monthly
           w in bounds, u_t>=0, z free
    """
    cols = list(returns.columns)

    df = returns.dropna(how="any")
    R = df.values  # T x N
    T, N = R.shape

    if T < 10:
        w_fallback = opt_min_variance(returns, bounds)
        return w_fallback, {"success": False, "message": "Too few rows for CVaR-cap", "fallback": "min_variance"}

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    mu = df.mean().values
    c = 1.0 / ((1.0 - alpha) * T)

    cap_m = float(cvar_cap_monthly)

    # Vars: [w_1..w_N, z, u_1..u_T]
    nvars = N + 1 + T

    # linprog minimizes => minimize -mu'w
    obj = np.zeros(nvars)
    obj[:N] = -mu

    # Inequalities
    # (1) -u_t - r_t'w - z <= 0
    # (2) z + c*sum(u_t) <= cap_m
    A_ub = np.zeros((T + 1, nvars))
    b_ub = np.zeros(T + 1)

    A_ub[:T, 0:N] = -R
    A_ub[:T, N] = -1.0
    A_ub[np.arange(T), N + 1 + np.arange(T)] = -1.0
    b_ub[:T] = 0.0

    A_ub[T, N] = 1.0
    A_ub[T, N + 1:] = c
    b_ub[T] = cap_m

    # Equality: sum(w)=1
    A_eq = np.zeros((1, nvars))
    A_eq[0, 0:N] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w in bounds, z free, u>=0
    var_bounds: list[tuple[float | None, float | None]] = []
    var_bounds.extend([(float(lo), float(hi)) for (lo, hi) in bounds])
    var_bounds.append((None, None))        # z
    var_bounds.extend([(0.0, None)] * T)   # u_t

    res = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=var_bounds,
        method="highs",
    )

    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback = opt_min_variance(returns, bounds)
        info = {
            "success": False,
            "message": str(getattr(res, "message", "")),
            "fallback": "min_variance",
            "alpha": alpha,
            "cvar_cap_monthly": cap_m,
        }
        return w_fallback, info

    w = res.x[:N]
    w = _clip_and_renormalize(w, bounds)

    info = {
        "success": True,
        "message": str(getattr(res, "message", "")),
        "fallback": "",
        "alpha": alpha,
        "cvar_cap_monthly": cap_m,
        "T": int(T),
    }
    return pd.Series(w, index=cols), info


# =========================================================


def tail_metrics(r: pd.Series, alpha: float = 0.90) -> pd.Series:
    x = r.dropna().astype(float)
    if len(x) == 0:
        return pd.Series(dtype=float)

    losses = -x.values
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else np.nan

    return pd.Series({
        f"VaR_{int(alpha*100)}_monthly": float(var),
        f"CVaR_{int(alpha*100)}_monthly": cvar,
        "worst_month": float(x.min()),
        "best_month": float(x.max()),
        "avg_month": float(x.mean()),
    })


