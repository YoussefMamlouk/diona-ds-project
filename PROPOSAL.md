Project Proposal
================

Title: Forecasting Equity Returns and Recommending Structured Products

Objective
---------
This project builds and evaluates a reproducible forecasting pipeline for equity prices and uses the forecasts to inform recommendations for structured products (e.g., autocallables, reverse convertibles, capital-protected notes). The primary research question is: "Can short- to medium-term forecasting models (ARIMA, AR(1), GARCH for volatility, and a machine learning baseline) produce actionable signals that help select historically superior structured-product strategies for a given risk tolerance and horizon?"

Data
----
Primary price data will be sourced from Yahoo Finance (via `yfinance`) with a Stooq fallback for resilience. For robustness and offline grading, the codebase includes a deterministic `--use-sample-data` mode that produces synthetic but reproducible daily data. Exogenous series (benchmarks, VIX, 10y-2y) will be fetched from Yahoo Finance or FRED (if API key provided).

Methods
-------
- Preprocessing: resample to the requested horizon (daily/weekly/monthly), compute log-returns and aligned exogenous variables; add a rolling 21-day historical volatility feature.
- Models: ARIMA (auto_arima), AR(1) with exogenous variables, naive Random Walk with drift, and an XGBoost baseline using engineered features.
- Volatility: GARCH(1,1) for volatility forecasting and Monte Carlo scenario generation combining return forecasts with GARCH-implied shocks.
- Evaluation: backtesting on the last historical window (MAE, RMSE, MAPE) to select the best-performing method per-horizon and to characterize signal quality.

Deliverables
------------
- Working repository with `python main.py` as the entry point and `--use-sample-data` for offline grading.
- Reproducible environment files (`requirements.txt`, `environment.yml`) and seeds set for deterministic behavior.
- Unit tests and a CI workflow for automated checks.
- A technical report describing methodology, results, and limitations; and an 8–10 minute presentation video demonstrating the code running.

Expected timeline
-----------------
Complete core implementation and tests (current state). Next: write the report and record the presentation, polish any packaging issues (optional), and optionally add integration tests gated by an environment variable.

