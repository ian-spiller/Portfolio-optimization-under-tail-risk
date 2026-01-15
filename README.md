# Portfolio Optimisation Under Tail Risk

This repository contains a collection of portfolio optimisation routines developed as a personal research project exploring how classical mean–variance allocation changes once tail-risk constraints are imposed.

The project compares several portfolio construction approaches using rolling historical windows:

- Classical minimum-variance portfolio  
- Target-return and target-volatility optimisation  
- Risk parity allocation  
- Utility-based optimisation  
- CVaR-based optimisation (including CVaR-max-return and CVaR-target-return formulations)

The primary motivation is to examine how downside-risk constraints reshape portfolio structure relative to standard Markowitz optimisation.

---

## Structure

`src/` contains modular optimisation routines implemented in Python.

`results/examples/` contains selected output files illustrating the behaviour of each optimisation method.

---

## Methods

The project implements rolling-window backtests of multiple portfolio construction techniques, including:

- Mean–variance optimisation  
- Risk parity  
- Sharpe-ratio maximisation  
- CVaR-constrained optimisation

Each method outputs portfolio weights and performance metrics, enabling direct comparison across frameworks.

---

## Motivation

Classical portfolio theory focuses on variance as the sole risk measure.  
This project explores how portfolio choice changes once downside structure is taken seriously, and whether traditional notions of optimality remain stable under tail-risk constraints.

---

## Status

This repository reflects an exploratory research project and is under continuous refinement.
