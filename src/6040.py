import numpy as np
import pandas as pd
import core

# --- settings ---
ALPHA = 0.90

TICKERS = {
    "SP500": "SPY",
    "UST_7_10Y": "IEF",
}

START = "2005-01-01"

# --- helper ---
def empirical_cvar(returns: pd.Series, alpha: float) -> float:
    losses = -returns.dropna().values
    var = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var]
    return float(tail_losses.mean())

# --- load data ---
rets = core.download_monthly_returns(TICKERS, START, None)

# --- 60/40 portfolio ---
w_6040 = pd.Series({"SP500": 0.6, "UST_7_10Y": 0.4})
port_rets = (rets * w_6040).sum(axis=1)

# --- CVaR ---
cvar_6040 = empirical_cvar(port_rets, ALPHA)

print(f"60/40 monthly CVaR (alpha={ALPHA:.0%}): {cvar_6040:.2%}")
