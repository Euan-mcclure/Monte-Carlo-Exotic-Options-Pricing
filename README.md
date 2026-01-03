# Monte-Carlo-Exotic-Options-Pricing

This project implements a Monte Carlo framework for pricing path-dependent exotic options (Asian, Barrier) under Black–Scholes. Variance reduction techniques are used to improve efficiency, and results are validated against analytical solutions and PDE benchmarks.

## Motivation

Pricing exotic options often lacks closed-form solutions. Monte Carlo simulation provides a flexible numerical method for path-dependent products. This project demonstrates:

- Efficient Monte Carlo simulation of GBM and stochastic volatility models
- Implementation of variance reduction techniques (antithetic, control variates)
- Benchmarking results against PDE solutions and analytical formulas

## Key Features

- Path-dependent options: arithmetic Asian, up-and-out barrier
- Variance reduction: antithetic variates, control variates
- Monte Carlo convergence analysis (number of paths, time step)
- PDE benchmarking (Crank–Nicolson finite difference)
- Greeks estimation (Delta, Vega) via pathwise and finite difference methods

