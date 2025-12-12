Project Proposal
================

Title: Machine Learning Model Comparison for Financial Asset Return and Volatility Forecasting

Objective
---------
This project implements and compares multiple forecasting models to predict financial asset returns and volatility over different investment horizons. The primary research question is:

**Which model best predicts asset returns and volatility over different investment horizons: statistical time-series models or machine learning models?**

The project explicitly evaluates whether machine learning models (XGBoost) outperform traditional statistical models (AR(1), ARIMA) and a simple baseline (Random Walk with Drift) for financial time series forecasting.

Data
----
Primary price data is sourced from Yahoo Finance (via `yfinance` library). For robustness and offline grading, the codebase includes a deterministic `--use-sample-data` mode that produces synthetic but reproducible data. Exogenous features include volume, momentum indicators, and rolling historical volatility.

Methods
-------
**Preprocessing:**
- Resample to requested horizon (daily/weekly/monthly)
- Compute log-returns and align exogenous variables
- Add rolling 21-day historical volatility feature

**Return Prediction Models:**
1. **Random Walk with Drift** (baseline) - Simple mean return model
2. **AR(1)** - Autoregressive model of order 1 with optional exogenous variables
3. **ARIMA** - Auto-selected ARIMA model (or fixed AR(1) for daily horizons)
4. **XGBoost Regressor** - Machine learning model with lag features and exogenous variables

**Volatility Prediction:**
- **GARCH(1,1)** - Generalized Autoregressive Conditional Heteroskedasticity model for forward-looking volatility forecasts

**Evaluation Methodology:**
- Time-series aware train/validation/test split (ensures no data leakage)
- Model selection based on validation set performance
- Final evaluation only on test set
- Metrics: RMSE, MAE, MAPE
- Explicit comparison against random walk baseline
- If ML models do not beat baseline, this is documented as a valid finding

**Monte Carlo Simulation:**
- Combines return forecasts with GARCH-implied volatility shocks
- Generates probabilistic price paths and confidence intervals

Deliverables
------------
- Working repository with `python main.py` as the entry point and `--use-sample-data` for offline grading
- Reproducible environment files (`requirements.txt`, `environment.yml`) with all dependencies
- Random seeds set everywhere (NumPy, sklearn, XGBoost) for deterministic behavior
- Model comparison outputs:
  - Model comparison CSV with RMSE, MAE, MAPE for all models
  - Price forecast plots with confidence intervals
  - Volatility forecast plots (historical vs GARCH)
  - Monte Carlo price path visualizations
- Unit tests for core functionality
- Technical report describing methodology, results, and limitations
- Presentation video demonstrating the code running

Key Findings
-----------
The project explicitly compares all models against the random walk baseline. The evaluation framework ensures:
- No data leakage through proper time-series splitting
- Fair comparison across all models
- Clear documentation of whether ML models outperform statistical models
- Recognition that baseline models are often strong for financial time series (efficient market hypothesis)

Expected Results
----------------
Given the efficient market hypothesis and the difficulty of forecasting financial time series, it is expected that:
- The random walk baseline may be competitive or even superior for some horizons
- AR(1) and ARIMA may show slight improvements for short horizons
- XGBoost may capture non-linear patterns but may overfit on limited financial data
- Results will vary by asset and horizon, demonstrating the importance of model comparison

This project provides a rigorous framework to evaluate these hypotheses and document findings transparently.
