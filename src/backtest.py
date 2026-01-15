"""Rolling backtest engine and portfolio-level utilities."""

from __future__ import annotations

import pandas as pd

from src.optimizers import build_bounds, validate_bounds


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
      meta: bounds metadata (min/max by asset)
      diag: diagnostic table (optimizer success, message, etc.)
    """
    if optimizer is None:
        raise ValueError("You must pass an optimizer callable.")

    optimizer_kwargs = optimizer_kwargs or {}

    # Optional asset-coverage filter (asset-level)
    if min_coverage < 1.0:
        cov = rets.notna().mean()
        keep = cov[cov >= float(min_coverage)].index.tolist()
        rets = rets[keep]

    # Drop rows with any NA (row-level)
    rets = rets.dropna(how="any")

    cols_all = list(rets.columns)

    if rf_col is not None:
        if rf_col not in cols_all:
            raise ValueError(f"rf_col='{rf_col}' not found in returns columns.")
        cols = [c for c in cols_all if c != rf_col]
    else:
        cols = cols_all

    bounds_list = build_bounds(cols, bounds_dict, default_bounds)
    validate_bounds(bounds_list)

    dates = rets.index
    if len(dates) <= lookback_months + 1:
        raise ValueError(
            f"Not enough monthly data after filtering. Need > {lookback_months+1}, got {len(dates)}."
        )

    weights_rows: list[pd.Series] = []
    oos_rows: list[pd.Series] = []
    diag_rows: list[dict] = []

    for i in range(lookback_months, len(dates) - 1):
        train = rets.iloc[i - lookback_months : i]
        test_date = dates[i + 1]
        test_ret = rets.iloc[i + 1]

        if rf_col is None:
            w, info = optimizer(train[cols], bounds_list, **optimizer_kwargs)
        else:
            rf_train = train[rf_col]
            w, info = optimizer(train[cols], rf_train, bounds_list, **optimizer_kwargs)

        w = w.reindex(cols).fillna(0.0)
        if float(w.sum()) == 0.0:
            w[:] = 1.0 / len(cols)
        w = w / float(w.sum())

        port_ret = float((w.values * test_ret[cols].values).sum())

        weights_rows.append(pd.Series(w.values, index=cols, name=test_date))
        oos_rows.append(pd.Series(port_ret, index=[test_date]))

        diag_rows.append({"date": test_date, **(info or {})})

    W = pd.DataFrame(weights_rows)

    # IMPORTANT: concat gives a Series already
    R = pd.concat(oos_rows).rename("oos_return")

    meta = pd.DataFrame(
        {
            "asset": cols,
            "min_weight": [b[0] for b in bounds_list],
            "max_weight": [b[1] for b in bounds_list],
        }
    )

    diag = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()
    return W, R, meta, diag


def apply_transaction_costs(
    weights: pd.DataFrame,
    oos_returns: pd.Series,
    cost_per_turnover: float = 0.0025,  # 25 bps per 1.0 turnover
) -> pd.DataFrame:
    """
    Computes gross vs net returns given portfolio weights and transaction costs.
    turnover_t = sum |w_t - w_{t-1}|
    """
    W = weights.copy()
    Rg = oos_returns.copy()

    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    costs = cost_per_turnover * turnover
    Rn = Rg - costs

    return pd.DataFrame(
        {
            "gross_return": Rg,
            "net_return": Rn,
            "turnover": turnover,
            "tcost": costs,
        }
    )

