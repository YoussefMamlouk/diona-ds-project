Project Proposal
================

Title: Return and Volatility Forecasting Across Investment Horizons

Objective
---------
This project implements a reproducible forecasting pipeline to compare statistical and machine-learning models for asset returns, and a dedicated volatility model for risk estimation. The primary research question is:

**Which modeling family best forecasts returns and volatility across short and long investment horizons?**

The code evaluates whether ML (XGBoost) provides measurable gains over statistical baselines, while treating volatility as a separate, more predictable target.

Data
----
Primary price data is sourced from Yahoo Finance (via `yfinance`) and cached for reproducibility. The default run uses cached TSLA data (fixed ticker) and does not download. Exogenous features include volume, momentum indicators, and rolling historical volatility.

Methods
-------
**Preprocessing:**
- Resample prices to the target horizon (daily/weekly/monthly)
- Compute log-returns and align exogenous variables
- Add a rolling 21-day historical volatility feature

**Return Prediction Models:**
1. **Random Walk with Drift** (baseline)
2. **AR(1)** with optional exogenous variables
3. **ARIMA** (fixed AR(1) for daily horizons; auto-selected otherwise)
4. **XGBoost Regressor** with lag and exogenous features
5. **Linear baselines** (Ridge/Lasso/ElasticNet, evaluation only)

**Volatility Prediction:**
- **GJR-GARCH(1,1)** with Student-t innovations on daily returns
- Forecasts are generated on daily data and aggregated to each horizon

**Evaluation Methodology:**
- Time-series train/validation/test split (no leakage)
- Model selection on validation; final metrics on test
- Return metrics: RMSE, MAE, MAPE
- Volatility backtest: rolling refit with forward realized variance target
- Volatility metrics: RMSE, MAE (annualized), and QLIKE
- Baseline for volatility: EWMA variance (lambda = 0.94)

**Monte Carlo Simulation:**
- Combines return forecasts with GARCH-driven volatility shocks
- Produces probabilistic price paths and confidence intervals

Horizon Design
--------------
The tool reports results on a fixed grid of horizons for comparability:
10 days, 1 month, 3 months, 6 months, 1 year.

Deliverables
------------
- Reproducible pipeline via `python main.py`
- Environment file: `environment.yml`
- Model comparison CSVs (validation and test)
- Price forecast plots with confidence bands
- Volatility forecast plots (historical vs GARCH)
- Volatility backtest summary table (GARCH vs EWMA baseline)
- EDA report and plots
- Technical report summarizing methodology, results, and limitations

Expected Results
----------------
Given the efficient market hypothesis:
- Simple baselines may remain competitive for returns
- ML gains may be limited to specific horizons or features
- Volatility should be more predictable than returns
- A properly specified GARCH model should add value over a naive EWMA baseline, especially at medium and longer horizons

This project focuses on transparent, reproducible comparisons rather than optimizing for a single headline metric.
