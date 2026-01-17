# Dynamic Portfolio Construction under Tail Risk
## Regime-Conditional Factor Risk & Scenario Simulation

---

## Overview

This project implements an end-to-end portfolio construction and risk analysis framework that explicitly separates **allocation decisions** from **risk evaluation**.

The objective is not return forecasting, but **understanding how portfolio risk behaves under stress, tail events, and changing market regimes**.

---

## Core Idea

Portfolio optimization determines an allocation; it does not fully describe the risks of that allocation.

This project treats optimization and risk analysis as complementary but distinct steps.  
Portfolios are built using standard rolling optimizers and then examined through factor-based simulations, regime-conditioned covariance, and explicit tail-risk decomposition.


Specifically, the project:

1. Constructs portfolios using classical rolling optimizers under realistic constraints  
2. Estimates time-varying macro factor exposures of the realized portfolio  
3. Simulates forward portfolio risk using fat-tailed factor distributions  
4. Conditions factor covariance on market regimes to reflect state-dependent risk

This structure preserves the strengths of traditional portfolio construction, while providing a more informative and economically interpretable view of tail risk and stress behavior.


---

## Implemented Components

### 1. Rolling Portfolio Optimizers

All optimizations are performed on a rolling monthly basis with realistic constraints and optional transaction costs.

Implemented optimizers:
- Maximum Sharpe Ratio (excess returns)
- Target Volatility
- Mean–Variance Utility
- CVaR-constrained return maximization

---

### 2. Factor Model

Portfolio returns are explained using macro factor proxies constructed from liquid ETFs:

- Risk-on / Risk-off
- Rates
- Credit
- Inflation
- USD

Rolling OLS regressions estimate **time-varying factor betas**, which serve as inputs for scenario simulation and tail-risk attribution.

---

### 3. Scenario Simulation

Forward one-month factor returns are simulated using:

- Multivariate Gaussian distribution
- Multivariate Student-t distribution (fat tails)

Portfolio returns are generated via a linear factor mapping:

\[
r_p = \beta^\top f
\]

Betas are taken **as-of the evaluation date** to avoid look-ahead bias.

---

### 4. Regime-Conditional Covariance

Covariance matrices are estimated conditionally on volatility regimes:

- Regimes are classified using rolling volatility of a chosen factor
- Separate covariance matrices are estimated for:
  - Low-volatility regime
  - High-volatility regime
- Simulations use the covariance corresponding to the **current regime**, with automatic fallback to unconditional estimates if data is insufficient

This captures the empirical behavior that correlations tend to increase during stress periods.

---

### 5. Tail Risk Decomposition

For simulated portfolio returns, the framework computes:

- Value-at-Risk (VaR) and Conditional VaR (CVaR)
- Probabilities of large losses
- Factor-level contributions to tail outcomes

This allows direct attribution of extreme losses to underlying macro risk drivers.

---

## Outputs

Each runner exports a structured Excel workbook containing:

- Rolling portfolio weights
- Out-of-sample portfolio returns
- Transaction cost diagnostics
- Factor regression results
- Rolling factor betas
- Scenario simulation summaries
- Tail-risk decomposition tables
- Regime diagnostics (covariance source, observation counts)

The output is designed to be transparent, auditable, and presentation-ready.

---

## Design Philosophy

- Portfolio optimization and risk evaluation are treated as **separate problems**
- All modeling assumptions are explicit
- No return forecasting or hidden alpha signals
- Modular design allows straightforward extensions

---

## Possible Extensions

The framework is intentionally modular and can be extended to:

- Multi-period scenario paths
- Copula-based dependence structures
- State-dependent expected returns
- Regime-aware portfolio optimization
- Futures and options overlays

---

## Disclaimer

This project is for research and educational purposes only.  
It does not constitute investment advice or a recommendation to trade.

---
