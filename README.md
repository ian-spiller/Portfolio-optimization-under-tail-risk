# Portfolio Allocation under Alternative Risk Definitions

This repository studies how the “optimal” portfolio allocation changes when we change the way risk is defined and constrained.

Instead of treating portfolio construction as a single optimization problem, the project compares multiple allocation paradigms that reflect how risk is handled in practice (variance-based vs tail-risk-based), under realistic constraints and implementation frictions.

All strategies are evaluated in a rolling, out-of-sample backtest framework using monthly data.

---

## Research Question

In applied asset management, “optimal allocation” depends on the chosen risk metric and the way risk budgets are imposed.

This project asks:

- How different are allocations when risk is measured by variance (Sharpe / utility / volatility targeting) versus tail risk (CVaR)?
- How stable are the resulting portfolios through time under rolling estimation?
- What is the cost of implementation once turnover and transaction costs are considered?

---

## Implemented Strategies

| Runner | Allocation rule | Risk definition / control |
|-------|------------------|---------------------------|
| run_rolling_markowitz.py | Max excess Sharpe | Variance-based (mean–variance) |
| run_target_vol.py | Max expected return subject to target volatility | Variance-based risk budget |
| run_utility.py | Max mean–variance utility (risk aversion λ) | Variance-based risk preference |
| run_cvar_max_return.py | Max expected return subject to CVaR(α) cap | Tail-risk constraint (CVaR) |

Each strategy uses the same evaluation protocol:

- Monthly rebalancing
- Rolling estimation window (train) and next-month evaluation (test)
- Strict allocation bounds
- Optional transaction-cost model based on turnover
- Identical data and reporting, enabling apples-to-apples comparison

---

## Tail-Risk Calibration

The CVaR cap is calibrated to an interpretable benchmark: the historical monthly CVaR(90%) of a classical 60/40 portfolio over the available sample.

This avoids arbitrary tail-risk budgets and makes the CVaR-constrained strategy comparable to a standard allocation baseline.

---

## Repository Structure

src/
  data.py        # data download & preprocessing
  optimizers.py  # portfolio construction engines
  backtest.py    # rolling out-of-sample evaluation
  reporting.py   # metrics & Excel export

run_*.py         # experiment runners  
results/         # generated outputs (not tracked in git)

---

## Outputs and Metrics

Each run exports an Excel workbook to `results/` containing:

- portfolio weights over time  
- out-of-sample returns (gross and net, if transaction costs enabled)  
- annualized return, volatility, Sharpe ratio  
- maximum drawdown  
- tail risk metrics (VaR, CVaR)  
- optimizer diagnostics (success / fallbacks)

---

## Usage

pip install -r requirements.txt

python run_rolling_markowitz.py  
python run_target_vol.py  
python run_utility.py  
python run_cvar_max_return.py


---

## Motivation

Most academic portfolio studies evaluate strategies under stylized assumptions.
This project focuses on practical portfolio construction under:

- alternative risk definitions (variance vs tail risk)
- realistic constraints
- rolling estimation error
- turnover and implementation costs

The goal is a clean, comparable framework for understanding how risk measurement choices shape allocation decisions.

