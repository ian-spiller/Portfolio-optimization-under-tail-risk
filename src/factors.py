# src/factors.py
"""Macro factor construction (Yahoo Finance ETF proxies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .data import download_monthly_returns


@dataclass(frozen=True)
class FactorSpec:
    """
    A factor built from either:
    - a single ticker (level return), e.g. USD = UUP
    - a long-short spread (ticker_long - ticker_short), e.g. RISK_ON = ACWI - BIL
    """
    name: str
    ticker: str | tuple[str, str]

    def required_tickers(self) -> list[str]:
        if isinstance(self.ticker, tuple):
            a, b = self.ticker
            return [a, b]
        return [self.ticker]


# -------------------------
# Frozen v1 macro factor set
# -------------------------
MACRO_FACTORS_V1: list[FactorSpec] = [
    FactorSpec("RISK_ON", ("ACWI", "SHY")),
    FactorSpec("RATES", ("TLT", "SHY")),
    FactorSpec("CREDIT", ("LQD", "IEF")),
    FactorSpec("INFLATION", ("TIP", "IEF")),
    FactorSpec("USD", "UUP"),
]


def _unique(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_factor_returns(
    specs: list[FactorSpec],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download tickers and construct monthly factor return series."""
    tickers = _unique([t for s in specs for t in s.required_tickers()])
    rets = download_monthly_returns(tickers=tickers, start=start, end=end)

    out = pd.DataFrame(index=rets.index)
    for s in specs:
        if isinstance(s.ticker, tuple):
            long_t, short_t = s.ticker
            out[s.name] = rets[long_t] - rets[short_t]
        else:
            out[s.name] = rets[s.ticker]

    out = out.dropna(how="any")
    if out.empty:
        raise ValueError("No factor data left after alignment / NA drops.")

    return out
