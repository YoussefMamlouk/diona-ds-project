# Machine Learning Model Comparison for Financial Asset Forecasting

## Project Overview

This project implements and compares multiple forecasting models to predict financial asset returns and volatility over different investment horizons. The research question is:

**Which model best predicts asset returns and volatility over different investment horizons: statistical time-series models or machine learning models?**

## Models Implemented

### Return Prediction Models
1. **Random Walk with Drift** (baseline) - Simple mean return model
2. **AR(1)** - Autoregressive model of order 1
3. **ARIMA** - Auto-selected ARIMA model (or fixed AR(1) for daily horizons)
4. **Ridge / Lasso / ElasticNet** - Regularized linear baselines (evaluation only)
5. **XGBoost Regressor** - Machine learning model with lag features and exogenous variables

**Note:** Linear baselines are reported in backtests for comparison; the production forecast uses Random Walk, AR(1), ARIMA, or XGBoost.

### Volatility Prediction
- **GJR-GARCH(1,1)** with Student-t innovations (fallback to standard GARCH if needed)
- **EWMA baseline** for volatility backtest comparison

## Evaluation Methodology

- **Time-series aware train/validation/test split** - Ensures no data leakage
- **Model selection** - Choose the best model on the **validation set** using **RMSE**
- **Final evaluation** - Refit the selected model on **train + validation**, then evaluate once on the **test set**
- **Return metrics**: RMSE, MAE, MAPE
- **Volatility metrics**: RMSE, MAE (annualized), QLIKE (variance loss)
- **Baseline comparison** - Explicit comparison against random walk baseline

## Key Findings

- **Returns**: After validation-based selection, no model consistently beats the random walk on the test set (the 1‑month case is a tie). This matches standard finance evidence that return predictability is weak.
- **Volatility**: GJR‑GARCH consistently outperforms the EWMA baseline across horizons, indicating volatility is more reliably forecastable than returns.
- **Diagnostics**: On the test set, ARIMA can win at 1 month and XGBoost at 3 months, but these wins are not used for selection to avoid test leakage.
- **Relative vs absolute**: We report absolute errors (RMSE/MAE) and interpret models relative to the random‑walk baseline, which is the appropriate benchmark for return forecasting.

## Setup

### Prerequisites
- Conda (required)

### Installation

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate stock-forecast
```

This automatically installs Python 3.10.0 and all dependencies.

## Usage

### Basic Usage

### Simple Usage

**Simply run the main file with no arguments:**

```bash
python main.py
```

This automatically runs:
- **Exploratory Data Analysis (EDA)** with comprehensive plots and insights
- **Forecasting for all horizons** (10 days, 1 month, 3 months, 6 months, 1 year)
- Uses **cached TSLA data** from `data/raw/` folder for reproducibility
- Saves outputs to the `results/` directory when `--save` is used

**Note:** The project is configured to use only TSLA (Tesla) data from the `data/raw/` folder for reproducibility. The ticker cannot be changed.

### Optional Flags

```
--save           Save plots/CSVs to results/ (off by default)
--no-ml          Disable XGBoost (faster, more stable on small samples)
--n-scenarios N  Monte Carlo scenarios (default 500)
```

### Supported Forecast Horizons
The CLI reports a fixed grid of horizons for comparability:
- **10 days**, **1 month**, **3 months**, **6 months**, **1 year**

## Outputs

The project generates the following outputs in the `results/` directory:

### Exploratory Data Analysis (EDA) Outputs

1. **Price Trend Analysis** (`eda_price_trend_<TICKER>_<TIMESTAMP>.png`)
   - Price movements with moving averages
   - Daily price ranges
   - Cumulative returns

2. **Returns Distribution** (`eda_returns_distribution_<TICKER>_<TIMESTAMP>.png`)
   - Histogram of returns
   - Q-Q plot for normality testing
   - Year-over-year comparisons
   - Time series of returns

3. **Volatility Analysis** (`eda_volatility_<TICKER>_<TIMESTAMP>.png`)
   - Rolling volatility (21-day and 252-day)
   - Volatility clustering patterns
   - Volatility distribution
   - Year-over-year volatility trends

4. **Volume Analysis** (`eda_volume_<TICKER>_<TIMESTAMP>.png`)
   - Trading volume trends
   - Price vs volume relationships
   - Volume distributions

5. **Correlation Analysis** (`eda_correlation_<TICKER>_<TIMESTAMP>.png`)
   - Correlation matrices
   - Returns vs volume relationships
   - Autocorrelation functions

6. **Statistical Summary** (`eda_statistics_<TICKER>_<TIMESTAMP>.txt`)
   - Comprehensive statistical metrics
   - Normality tests
   - Risk assessments

### Forecasting Outputs

1. **Price Forecast Plot** (`forecast_<TICKER>_<HORIZON>_<TIMESTAMP>.png`)
   - Historical prices
   - Forecasted prices with confidence intervals
   - Monte Carlo percentiles (10th, 50th, 90th)

2. **Volatility Forecast Plot** (`volatility_forecast_<TICKER>_<HORIZON>_<TIMESTAMP>.png`)
   - Historical rolling volatility
   - GARCH-based (GJR-GARCH) forecasted volatility

3. **Model Comparison CSVs**
   - Validation: `model_comparison_validation_<TICKER>_<HORIZON>_<TIMESTAMP>.csv`
   - Test: `model_comparison_test_<TICKER>_<HORIZON>_<TIMESTAMP>.csv`
   - RMSE, MAE, MAPE for each model
   - Indicates whether ML models beat the baseline

## Data

- **Data source**: Cached TSLA (Tesla) data from `data/raw/`
- **Reproducibility**: Default run uses only cached data (no downloads)
- **Offline capability**: Works entirely offline using cached data
- **No API keys required**: The project does not use any external APIs or require API keys
- **Fixed ticker**: The project is configured to work only with TSLA data for reproducibility

## Project Structure

```
.
├── main.py                 # Main entry point
├── src/
│   ├── data_loader.py     # Data fetching and preprocessing
│   ├── models.py          # Model implementations (AR(1), ARIMA, XGBoost, linear baselines)
│   ├── backtests.py       # Return backtests and model selection
│   ├── volatility.py      # Volatility backtests (GARCH vs EWMA)
│   ├── plots.py           # Forecast and volatility plots
│   ├── results.py         # CSV output and results cleanup
│   └── evaluation.py      # Forecasting orchestration (returns + volatility)
├── data/
│   └── raw/               # Raw data storage
├── results/               # Generated outputs (plots, CSVs)
├── notebooks/             # Exploratory notebooks
└── environment.yml        # Conda environment file
```

## Reproducibility

The project is designed for maximum reproducibility:
- **Fixed ticker**: Always uses TSLA (Tesla) data from `data/raw/` folder
- **Cached data only**: Uses the fixed CSV snapshot in `data/raw/` (no downloads)
- **Pinned package versions**: All package versions are pinned to exact versions in `environment.yml` to ensure consistent results across devices
- **Random seeds**: All random seeds are set to ensure reproducibility:
  - NumPy: `np.random.seed(42)` (set before all model training)
  - XGBoost: `random_state=42`
  - Auto ARIMA: `random_state=42, n_jobs=1`
  - Monte Carlo: `np.random.seed(42)`
  - GARCH fitting: seed set before optimization
- **Environment variables**: Set for numerical reproducibility:
  - `PYTHONHASHSEED=0` for hash-based operations
  - `OMP_NUM_THREADS=1` and related threading vars for single-threaded BLAS
- **No user input**: No prompts or interactive elements - fully deterministic execution

### Verifying Package Versions

To ensure your environment matches the expected versions, run:

```bash
python verify_versions.py
```

This script checks that all installed packages match the pinned versions in `environment.yml`.

### Setting Up the Environment

**⚠️ IMPORTANT: This project requires Python 3.10.0 for reproducibility.**

**Using conda:**
```bash
conda env create -f environment.yml
conda activate stock-forecast
```

This automatically installs Python 3.10.0 and all dependencies as specified in `environment.yml`.

**Why Python 3.10.0?**
- Matches the exact version specified in `environment.yml`
- Prevents version-related discrepancies in numerical computations
- Ensures maximum reproducibility across all devices

## Key Findings

The project explicitly compares all models against the random walk baseline. If machine learning models do not always outperform the baseline, this is documented as a valid finding (which is common in financial time series forecasting).

## License

This project is for educational purposes.

## Contact

For questions or issues, please refer to the project documentation or contact the maintainer.
