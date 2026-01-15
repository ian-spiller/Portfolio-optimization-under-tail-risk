"""
Optimisation routines and shared helpers.

Contains:
- bounds utilities
- covariance stabilisation + weight post-processing
- a set of pluggable portfolio optimisers (mean-variance, target-vol, risk parity, CVaR, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog


def build_bounds(
    columns: list[str],
    bounds_dict: dict[str, tuple[float, float]],
    default_bounds: tuple[float, float] = (0.0, 1.0),
) -> list[tuple[float, float]]:
    return [bounds_dict.get(c, default_bounds) for c in columns]


def validate_bounds(bounds: list[tuple[float, float]]) -> None:
    for lo, hi in bounds:
        if lo > hi:
            raise ValueError(f"Invalid bounds: {(lo, hi)}")
        if hi < 0:
            raise ValueError(f"Upper bound < 0: {(lo, hi)}")


def _cov_stable(X: np.ndarray, shrink: float = 1e-6) -> np.ndarray:
    return np.cov(X, rowvar=False) + shrink * np.eye(X.shape[1])


def _clip_and_renormalize(w: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Enforce bounds and sum-to-1 in a robust way.
    """
    w2 = w.copy()
    for i, (lo, hi) in enumerate(bounds):
        w2[i] = np.clip(w2[i], lo, hi)

    s = float(w2.sum())
    if s <= 0:
        # fallback: put everything into the first asset with max bound > 0
        w2[:] = 0.0
        for i, (lo, hi) in enumerate(bounds):
            if hi > 0:
                w2[i] = min(1.0, hi)
                break
        s = float(w2.sum())

    w2 /= s
    return w2


def opt_max_sharpe_excess(
    returns: pd.DataFrame,
    rf: pd.Series,
    bounds: list[tuple[float, float]],
) -> tuple[pd.Series, dict]:
    """
    Maximize Sharpe of excess returns (monthly) using mean/cov.
    """
    cols = list(returns.columns)
    df = returns.dropna(how="any")
    rf = rf.reindex(df.index).dropna()
    df = df.reindex(rf.index).dropna(how="any")

    X = df.values
    mu = df.mean().values
    cov = _cov_stable(X)

    # Excess mean
    mu_ex = mu - float(rf.mean())

    N = len(cols)

    def neg_sharpe(w):
        w = np.asarray(w)
        ret = float(w @ mu_ex)
        vol = float(np.sqrt(w @ cov @ w))
        if vol <= 1e-12:
            return 1e6
        return -ret / vol

    w0 = np.ones(N) / N
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=[cons], method="SLSQP")
    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(df, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = _clip_and_renormalize(res.x, bounds)
    info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}
    return pd.Series(w, index=cols), info


def opt_min_variance(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
) -> tuple[pd.Series, dict]:
    """
    Minimize portfolio variance.
    """
    cols = list(returns.columns)
    df = returns.dropna(how="any")
    X = df.values
    cov = _cov_stable(X)

    N = len(cols)

    def obj(w):
        w = np.asarray(w)
        return float(w @ cov @ w)

    w0 = np.ones(N) / N
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(obj, w0, bounds=bounds, constraints=[cons], method="SLSQP")
    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        # equal weights within bounds
        w = np.ones(N) / N
        w = _clip_and_renormalize(w, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "equal_weight"}
        return pd.Series(w, index=cols), info

    w = _clip_and_renormalize(res.x, bounds)
    info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": ""}
    return pd.Series(w, index=cols), info


def opt_mean_variance_utility(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    risk_aversion: float = 5.0,
) -> tuple[pd.Series, dict]:
    """
    Maximize mean - (risk_aversion/2)*variance (monthly stats).
    """
    cols = list(returns.columns)
    df = returns.dropna(how="any")
    X = df.values
    mu = df.mean().values
    cov = _cov_stable(X)

    N = len(cols)

    def neg_utility(w):
        w = np.asarray(w)
        m = float(w @ mu)
        v = float(w @ cov @ w)
        return -(m - 0.5 * float(risk_aversion) * v)

    w0 = np.ones(N) / N
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_utility, w0, bounds=bounds, constraints=[cons], method="SLSQP")
    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(df, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = _clip_and_renormalize(res.x, bounds)
    info = {"success": True, "message": str(getattr(res, "message", "")), "fallback": "", "risk_aversion": risk_aversion}
    return pd.Series(w, index=cols), info


def opt_target_vol(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    target_vol_annual: float = 0.12,
) -> tuple[pd.Series, dict]:
    """
    Find weights with minimum variance subject to annualized vol ~ target_vol_annual.
    Implemented as min variance with equality constraint on vol (SLSQP).
    """
    cols = list(returns.columns)
    df = returns.dropna(how="any")
    X = df.values
    cov = _cov_stable(X)

    N = len(cols)
    target_m = float(target_vol_annual) / np.sqrt(12.0)

    def variance(w):
        w = np.asarray(w)
        return float(w @ cov @ w)

    def vol_constraint(w):
        w = np.asarray(w)
        vol = float(np.sqrt(w @ cov @ w))
        return vol - target_m

    w0 = np.ones(N) / N
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": vol_constraint},
    ]

    res = minimize(variance, w0, bounds=bounds, constraints=cons, method="SLSQP")
    if (not res.success) or (res.x is None) or np.any(~np.isfinite(res.x)):
        w_fallback, _ = opt_min_variance(df, bounds)
        info = {"success": False, "message": str(getattr(res, "message", "")), "fallback": "min_variance"}
        return w_fallback, info

    w = _clip_and_renormalize(res.x, bounds)
    info = {
        "success": True,
        "message": str(getattr(res, "message", "")),
        "fallback": "",
        "target_vol_annual": target_vol_annual,
        "target_vol_monthly": target_m,
    }
    return pd.Series(w, index=cols), info

def opt_cvar_cap_max_return_monthly(
    returns: pd.DataFrame,
    bounds: list[tuple[float, float]],
    alpha: float = 0.95,
    cvar_cap_monthly: float = 0.035,   # cap on MONTHLY CVaR (loss)
) -> tuple[pd.Series, dict]:
    """
    Same as opt_cvar_cap_max_return, but you provide the cap directly in MONTHLY terms.
    """
    cols = list(returns.columns)

    df = returns.dropna(how="any")
    X = df.values
    T, N = X.shape

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    mu = df.mean().values  # monthly expected returns
    c = 1.0 / ((1.0 - alpha) * T)

    cap_m = float(cvar_cap_monthly)

    # Decision variables: [w_1..w_N, z, u_1..u_T] => N + 1 + T
    nvars = N + 1 + T

    # Objective: maximize mu'w -> minimize -mu'w
    obj = np.zeros(nvars)
    obj[:N] = -mu

    # Equality: sum(w)=1
    Aeq = np.zeros((1, nvars))
    Aeq[0, :N] = 1.0
    beq = np.array([1.0])

    # Inequalities:
    # 1) -u_t - r_t'w - z <= 0
    # 2) z + c*sum(u_t) <= cap_m
    A = np.zeros((T + 1, nvars))
    b = np.zeros(T + 1)

    for t in range(T):
        A[t, :N] = -X[t, :]
        A[t, N] = -1.0
        A[t, N + 1 + t] = -1.0
        b[t] = 0.0

    # CVaR cap constraint
    A[T, N] = 1.0
    A[T, N + 1 :] = c
    b[T] = cap_m

    var_bounds = []
    for lo, hi in bounds:
        var_bounds.append((lo, hi))
    var_bounds.append((None, None))  # z free
    for _ in range(T):
        var_bounds.append((0.0, None))  # u_t >= 0

    validate_bounds(bounds)

    res = linprog(
        c=obj,
        A_ub=A,
        b_ub=b,
        A_eq=Aeq,
        b_eq=beq,
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
        "cap_monthly": cap_m,
        "T": int(T),
    }
    return pd.Series(w, index=cols), info
