"""
Data utilities for the portfolio-optimisation project.

This module provides helpers to download and prepare monthly return series.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_monthly_returns(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted prices from yfinance and convert to MONTHLY returns.

    Returns: DataFrame of monthly returns (end-of-month), columns=tickers
    """
    yf_tickers = " ".join(tickers)
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

    px = px.dropna(how="all")

    # Monthly prices (last business day in month)
    px_m = px.resample("M").last()

    rets_m = px_m.pct_change().dropna(how="all")
    rets_m.index = pd.to_datetime(rets_m.index)

    # Ensure columns in same order
    rets_m = rets_m.reindex(columns=tickers)

    return rets_m


def prepare_returns(
    rets: pd.DataFrame,
    min_coverage: float = 1.0,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cleans the returns matrix:
    - drops assets with insufficient history coverage (fraction of months with non-NA)
    - forces keeping columns in keep_cols (if provided)
    - drops rows with any missing values after filtering
    """
    if rets.empty:
        raise ValueError("Returns DataFrame is empty.")

    keep_cols = keep_cols or []

    cov = rets.notna().mean()

    # columns that pass coverage OR are forced to be kept
    keep = set(cov[cov >= float(min_coverage)].index.tolist()) | set(keep_cols)
    keep = [c for c in rets.columns if c in keep]

    out = rets[keep].copy()

    # Now drop rows with any NA (so optimizers see a clean matrix)
    out = out.dropna(how="any")
    if out.empty:
        raise ValueError("No data left after filtering for min_coverage / NA rows.")

    return out

