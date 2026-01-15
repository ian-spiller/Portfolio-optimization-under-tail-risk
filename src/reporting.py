"""Reporting helpers (performance summary, tail metrics, Excel export)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def perf_summary(oos_returns: pd.Series) -> pd.Series:
    r = oos_returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float)

    ann_return = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan
    max_dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()

    return pd.Series(
        {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "months": len(r),
        }
    )


def tail_metrics(r: pd.Series, alpha: float = 0.95) -> pd.Series:
    x = r.dropna().astype(float).values
    if x.size == 0:
        return pd.Series(dtype=float)

    q = np.quantile(x, 1 - alpha)  # VaR at (1-alpha)
    es = x[x <= q].mean() if np.any(x <= q) else np.nan
    return pd.Series({"VaR": q, "CVaR": es})


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
                df.to_excel(writer, sheet_name=str(name)[:31], index=True)
