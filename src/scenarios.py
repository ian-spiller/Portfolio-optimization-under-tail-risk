# src/scenarios.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


Method = Literal["gaussian", "student_t"]
RegimeMethod = Literal["none", "rolling_vol"]
RegimeName = Literal["current", "low", "high"]


@dataclass(frozen=True)
class SimulationSpec:
    """
    Defines how we simulate factor returns.

    - gaussian:   f ~ N(mu, Sigma)
    - student_t:  f ~ multivariate Student-t with df degrees of freedom
                 (implemented as correlated normal scaled by chi-square shock)
    """
    method: Method = "gaussian"
    n_sims: int = 20_000
    horizon_months: int = 1               # next-month stress by default
    lookback_months: int = 60             # estimation window for mu/Sigma
    seed: Optional[int] = 42
    demean: bool = True                   # risk-focused shocks (mu=0); not alpha forecasting
    df: int = 6                           # ONLY for student_t (must be > 2 for finite variance)


@dataclass(frozen=True)
class RegimeSpec:
    """
    Optional regime-conditional covariance.

    method:
      - "none": unconditional covariance (default)
      - "rolling_vol": classify regimes by rolling volatility of a chosen factor

    regime:
      - "current": use the regime at the last available date in the estimation sample
      - "low": force low-vol regime
      - "high": force high-vol regime
    """
    method: RegimeMethod = "none"
    factor_for_vol: str = "RISK_ON"
    vol_window: int = 12
    threshold_quantile: float = 0.7       # top 30% vol is "high"
    regime: RegimeName = "current"
    min_obs_per_regime: int = 24          # fallback to unconditional if fewer observations


@dataclass(frozen=True)
class TailSpec:
    """
    Defines tail metrics to report.
    """
    alpha: float = 0.95
    loss_thresholds: tuple[float, ...] = (0.02, 0.05)


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Optional deterministic stress scenario in factor space.
    Example: {"RISK_ON": -0.08, "RATES": +0.02, "CREDIT": -0.03, "INFLATION": +0.01, "USD": +0.02}
    """
    name: str
    factor_shocks: dict[str, float]


@dataclass
class ScenarioReport:
    """
    Returned by run_factor_scenarios(). Designed to be directly exportable to Excel sheets.
    """
    sim_returns: pd.Series
    summary: pd.DataFrame
    tail_decomp: pd.DataFrame
    stress_table: Optional[pd.DataFrame] = None


# ============================================================
# 1) Core helpers (public)
# ============================================================

def pick_beta_vector(
    betas: pd.DataFrame,
    factor_names: list[str],
    asof: Optional[pd.Timestamp] = None,
    method: Literal["latest", "asof"] = "latest",
    const_name: str = "const",
) -> pd.Series:
    """
    Convert rolling betas DataFrame into a single beta vector.

    Keeps ONLY the requested factor_names (drops alpha/r2/extra diagnostics, etc.).
    """
    if betas.empty:
        raise ValueError("betas is empty")

    if method == "latest" or asof is None:
        row = betas.iloc[-1]
    else:
        sub = betas.loc[:asof]
        if sub.empty:
            raise ValueError(f"No beta row available on/before {asof}")
        row = sub.iloc[-1]

    # drop intercept columns robustly
    row = row.drop(labels=[const_name, "Intercept", "const"], errors="ignore")

    # keep ONLY factor betas, in the same order as factor_returns columns
    row = row.reindex(factor_names)

    if row.isna().any():
        missing = list(row.index[row.isna()])
        raise ValueError(f"Missing beta estimates for factors: {missing}")

    return row.astype(float)


def _label_regimes_rolling_vol(
    F: pd.DataFrame,
    factor_for_vol: str,
    vol_window: int,
    threshold_quantile: float,
) -> pd.Series:
    if factor_for_vol not in F.columns:
        raise ValueError(f"Regime factor_for_vol='{factor_for_vol}' not in factor_returns columns")

    vol = F[factor_for_vol].rolling(vol_window).std()
    thr = float(vol.quantile(threshold_quantile))
    regime = pd.Series(np.where(vol > thr, "high", "low"), index=F.index, name="regime")
    return regime


def _select_regime_sample(
    F: pd.DataFrame,
    regime_spec: RegimeSpec,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      F_used: DataFrame used to estimate mu/cov
      info: dict with regime diagnostics (used in summary)
    """
    info = {
        "regime_method": regime_spec.method,
        "regime_factor_for_vol": regime_spec.factor_for_vol,
        "regime_vol_window": regime_spec.vol_window,
        "regime_threshold_q": regime_spec.threshold_quantile,
        "regime_requested": regime_spec.regime,
        "regime_used": "unconditional",
        "regime_obs": len(F),
        "regime_fallback": False,
    }

    if regime_spec.method == "none":
        return F, info

    # rolling_vol regime
    labels = _label_regimes_rolling_vol(
        F,
        factor_for_vol=regime_spec.factor_for_vol,
        vol_window=regime_spec.vol_window,
        threshold_quantile=regime_spec.threshold_quantile,
    )

    if regime_spec.regime == "current":
        target = str(labels.iloc[-1])
    else:
        target = str(regime_spec.regime)

    F_reg = F.loc[labels == target].dropna(how="any")

    info["regime_used"] = target
    info["regime_obs"] = len(F_reg)

    if len(F_reg) < regime_spec.min_obs_per_regime:
        # fallback: unconditional if too few observations
        info["regime_used"] = "fallback_unconditional"
        info["regime_obs"] = len(F)
        info["regime_fallback"] = True
        return F, info

    return F_reg, info


def simulate_factor_returns(
    factor_returns: pd.DataFrame,
    spec: SimulationSpec,
    end_date: Optional[pd.Timestamp] = None,
    regime: Optional[RegimeSpec] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate factor returns over horizon_months using either:
      - Gaussian multivariate normal
      - Multivariate Student-t (fat tails)

    If `regime` is provided, estimate mu/cov using a regime-conditional subset.

    Returns
    -------
    (sim_factors, sim_info)

    sim_factors:
      For horizon_months=1: shape (n_sims, n_factors)
      For horizon_months>1: MultiIndex DataFrame with index=(sim_id, step)

    sim_info:
      dict with estimation diagnostics (used regime, obs count, etc.)
    """
    F = factor_returns.copy()

    if end_date is not None:
        F = F.loc[:end_date]

    if len(F) < spec.lookback_months:
        raise ValueError(
            f"Not enough factor history: have {len(F)} rows, need >= {spec.lookback_months}"
        )

    # base estimation window
    F = F.iloc[-spec.lookback_months:]
    F = F.dropna(how="any")
    if F.empty:
        raise ValueError("All factor rows are NaN after dropping missing values")

    # regime selection (optional)
    regime = regime or RegimeSpec(method="none")
    F_used, reg_info = _select_regime_sample(F, regime)

    rng = np.random.default_rng(spec.seed)

    X = F_used.to_numpy(dtype=float)
    if spec.demean:
        X = X - X.mean(axis=0, keepdims=True)

    n_factors = X.shape[1]

    # parameters from chosen sample
    mu = np.zeros(n_factors) if spec.demean else F_used.mean().to_numpy(dtype=float)
    cov = np.cov(X, rowvar=False)

    sim_info = {
        "sim_method": spec.method,
        "sim_df": int(spec.df) if spec.method == "student_t" else np.nan,
        "sim_lookback_months": int(spec.lookback_months),
        "sim_used_obs": int(len(F_used)),
        **reg_info,
    }

    if spec.method == "gaussian":
        if spec.horizon_months == 1:
            sims = rng.multivariate_normal(mean=mu, cov=cov, size=spec.n_sims)
            return pd.DataFrame(sims, columns=F.columns), sim_info

        sims = rng.multivariate_normal(mean=mu, cov=cov, size=(spec.n_sims, spec.horizon_months))
        arr = sims.reshape(spec.n_sims * spec.horizon_months, n_factors)
        mi = pd.MultiIndex.from_product(
            [range(spec.n_sims), range(spec.horizon_months)],
            names=["sim_id", "step"],
        )
        return pd.DataFrame(arr, index=mi, columns=F.columns), sim_info

    if spec.method == "student_t":
        df = int(spec.df)
        if df <= 2:
            raise ValueError("student_t requires df > 2 for finite variance")

        if spec.horizon_months == 1:
            z = rng.multivariate_normal(mean=np.zeros(n_factors), cov=cov, size=spec.n_sims)
            g = rng.chisquare(df, size=spec.n_sims) / df
            sims = mu + z / np.sqrt(g)[:, None]
            return pd.DataFrame(sims, columns=F.columns), sim_info

        z = rng.multivariate_normal(
            mean=np.zeros(n_factors),
            cov=cov,
            size=(spec.n_sims, spec.horizon_months),
        )
        g = rng.chisquare(df, size=(spec.n_sims, spec.horizon_months)) / df
        sims = mu + z / np.sqrt(g)[..., None]

        arr = sims.reshape(spec.n_sims * spec.horizon_months, n_factors)
        mi = pd.MultiIndex.from_product(
            [range(spec.n_sims), range(spec.horizon_months)],
            names=["sim_id", "step"],
        )
        return pd.DataFrame(arr, index=mi, columns=F.columns), sim_info

    raise ValueError(f"Unknown method: {spec.method}")


def map_factors_to_portfolio_returns(
    sim_factors: pd.DataFrame,
    beta: pd.Series,
    include_intercept: bool = False,
    intercept: float = 0.0,
) -> pd.Series:
    """
    Convert simulated factor returns into simulated portfolio returns via linear mapping.

      r_p = beta' f  (+ intercept if enabled)
    """
    missing = [c for c in beta.index if c not in sim_factors.columns]
    if missing:
        raise ValueError(f"sim_factors missing factor columns: {missing}")

    X = sim_factors[beta.index].to_numpy(dtype=float)
    b = beta.to_numpy(dtype=float)
    rp = X @ b
    if include_intercept:
        rp = rp + float(intercept)
    return pd.Series(rp, name="sim_return")


def compute_tail_summary(sim_returns: pd.Series, tail: TailSpec) -> pd.DataFrame:
    """
    Summary table for simulated returns.
    Convention: left tail = bad. We compute VaR/CVaR on LOSSES.
    """
    r = sim_returns.dropna().astype(float)
    if r.empty:
        raise ValueError("sim_returns is empty")

    losses = -r
    var = float(np.quantile(losses, tail.alpha))
    cvar = float(losses[losses >= var].mean())

    rows = {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        f"VaR_{int(tail.alpha*100)}_loss": var,
        f"CVaR_{int(tail.alpha*100)}_loss": cvar,
        "min": float(r.min()),
        "max": float(r.max()),
    }

    for th in tail.loss_thresholds:
        rows[f"P(loss>{th:.0%})"] = float((losses > th).mean())

    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


def tail_decomposition(
    sim_factors: pd.DataFrame,
    beta: pd.Series,
    sim_returns: pd.Series,
    tail: TailSpec,
) -> pd.DataFrame:
    """
    Decompose tail outcomes by factor contributions.

      contribution_i = beta_i * factor_i
    """
    r = sim_returns.astype(float)
    losses = -r
    var = float(np.quantile(losses, tail.alpha))
    tail_mask = losses >= var

    contrib = sim_factors[beta.index].mul(beta, axis=1)
    contrib_tail = contrib.loc[tail_mask]

    out = pd.DataFrame({
        "mean_contrib_in_tail": contrib_tail.mean(),
        "median_contrib_in_tail": contrib_tail.median(),
        "p10_contrib_in_tail": contrib_tail.quantile(0.10),
        "p90_contrib_in_tail": contrib_tail.quantile(0.90),
    })

    total_tail = contrib_tail.sum(axis=1)
    denom = total_tail.replace(0.0, np.nan)
    out["mean_share_in_tail"] = (contrib_tail.div(denom, axis=0)).mean()

    return out.sort_values("mean_contrib_in_tail")


def run_factor_scenarios(
    factor_returns: pd.DataFrame,
    rolling_betas: pd.DataFrame,
    asof: Optional[pd.Timestamp] = None,
    beta_pick: Literal["latest", "asof"] = "latest",
    sim: SimulationSpec = SimulationSpec(),
    regime: Optional[RegimeSpec] = None,
    tail: TailSpec = TailSpec(),
    stress_scenarios: Optional[list[ScenarioSpec]] = None,
    include_intercept: bool = False,
) -> ScenarioReport:
    """
    One-stop function you call from your runners.

    Supports Gaussian or Student-t factor simulation.
    Optional regime-conditional covariance.
    """
    beta = pick_beta_vector(
        rolling_betas,
        factor_names=list(factor_returns.columns),
        asof=asof,
        method=beta_pick,
    )
    end_date = asof if beta_pick == "asof" else None

    sim_f, sim_info = simulate_factor_returns(
        factor_returns,
        spec=sim,
        end_date=end_date,
        regime=regime,
    )
    sim_r = map_factors_to_portfolio_returns(sim_f, beta, include_intercept=include_intercept)

    summary = compute_tail_summary(sim_r, tail)

    # attach simulation/regime diagnostics into summary (nice for Excel)
    for k, v in sim_info.items():
        summary.loc[f"diag_{k}", "value"] = v

    decomp = tail_decomposition(sim_f, beta, sim_r, tail)

    stress_tbl = None
    if stress_scenarios:
        rows = []
        for sc in stress_scenarios:
            f = pd.Series(sc.factor_shocks, dtype=float).reindex(beta.index).fillna(0.0)
            contrib = f * beta
            rp = float(contrib.sum())
            row = {"scenario": sc.name, "portfolio_return": rp}
            for kk, vv in contrib.items():
                row[f"contrib_{kk}"] = float(vv)
            rows.append(row)
        stress_tbl = pd.DataFrame(rows).set_index("scenario")

    return ScenarioReport(
        sim_returns=sim_r,
        summary=summary,
        tail_decomp=decomp,
        stress_table=stress_tbl,
    )
