# src/factor_model.py
"""OLS factor regression + rolling exposures (monthly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _align(port_ret: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    r = port_ret.astype(float).copy()
    f = factors.astype(float).copy()

    df = pd.concat([r.rename("port"), f], axis=1).dropna(how="any")
    if df.empty:
        raise ValueError("No overlapping data between portfolio returns and factors.")
    return df["port"], df.drop(columns=["port"])


def ols_regression(
    port_ret: pd.Series,
    factors: pd.DataFrame,
    add_const: bool = True,
) -> dict[str, pd.DataFrame]:
    """port_ret = alpha + betas' * factors + eps"""
    y, X = _align(port_ret, factors)

    yv = y.values.reshape(-1, 1)
    Xv = X.values

    if add_const:
        Xv = np.column_stack([np.ones(len(Xv)), Xv])
        names = ["alpha"] + list(X.columns)
    else:
        names = list(X.columns)

    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    coef = coef.flatten()

    n = Xv.shape[0]
    k = Xv.shape[1]

    yhat = Xv @ coef
    resid = y.values - yhat

    sse = float(np.sum(resid**2))
    tss = float(np.sum((y.values - y.values.mean()) ** 2))
    r2 = 1.0 - sse / tss if tss > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if (n - k) > 0 and not np.isnan(r2) else np.nan

    dof = n - k
    sigma2 = sse / dof if dof > 0 else np.nan

    xtx_inv = np.linalg.inv(Xv.T @ Xv) if dof > 0 else np.full((k, k), np.nan)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    t_stats = coef / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof) if dof > 0 else np.full_like(t_stats, np.nan)

    coef_df = pd.DataFrame(
        {"coef": coef, "stderr": se, "t": t_stats, "p": p_vals},
        index=names,
    )

    fit_df = pd.DataFrame(
        {"value": {"r2": r2, "adj_r2": adj_r2, "nobs": n, "resid_std": float(np.sqrt(sigma2)) if sigma2 == sigma2 else np.nan}}
    )

    return {"coef": coef_df, "fit": fit_df}


def rolling_ols(
    port_ret: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
    add_const: bool = True,
) -> pd.DataFrame:
    """Rolling OLS betas (window in months). Returns alpha/betas + r2."""
    y, X = _align(port_ret, factors)

    rows: list[pd.Series] = []
    idx: list[pd.Timestamp] = []

    for i in range(window, len(y) + 1):
        y_w = y.iloc[i - window : i]
        X_w = X.iloc[i - window : i]
        res = ols_regression(y_w, X_w, add_const=add_const)

        row = res["coef"]["coef"].copy()
        row.loc["r2"] = float(res["fit"].loc["r2", "value"])

        rows.append(row)
        idx.append(y.index[i - 1])

    return pd.DataFrame(rows, index=pd.to_datetime(idx))
