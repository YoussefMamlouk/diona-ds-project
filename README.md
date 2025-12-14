# Machine Learning Model Comparison for Financial Asset Forecasting

## Project Overview

This project implements and compares multiple forecasting models to predict financial asset returns and volatility over different investment horizons. The research question is:

**Which model best predicts asset returns and volatility over different investment horizons: statistical time-series models or machine learning models?**

## Models Implemented

### Return Prediction Models
1. **Random Walk with Drift** (baseline) - Simple mean return model
2. **AR(1)** - Autoregressive model of order 1
3. **ARIMA** - Auto-selected ARIMA model (or fixed AR(1) for daily horizons)
4. **XGBoost Regressor** - Machine learning model with lag features and exogenous variables

### Volatility Prediction
- **GARCH(1,1)** - Generalized Autoregressive Conditional Heteroskedasticity model for volatility forecasting

## Evaluation Methodology

- **Time-series aware train/validation/test split** - Ensures no data leakage
- **Model selection** - Based on validation set performance
- **Final evaluation** - Only on test set
- **Metrics**: RMSE, MAE, MAPE
- **Baseline comparison** - Explicit comparison against random walk baseline

## Setup

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or pip

### Installation

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate stock-forecast
```

**Using pip:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

The script will prompt you for:
- Ticker symbol (e.g., TSLA, AAPL)
- Investment horizon (e.g., "10 days", "3 months", "1 year")

### Command Line Options

```bash
# Run ALL forecast horizons in a single run (recommended)
python main.py --demo --all-horizons --save
# This runs: 10 days, 1 month, 3 months, 6 months, 1 year
# Generates comprehensive comparison across all timeframes

# Run deterministic demo with sample data (for offline grading)
# Defaults to 10 days, but can specify longer horizons
python main.py --demo --use-sample-data --save
python main.py --demo --horizon "3 months" --save  # Override default with longer horizon

# Run with specific horizon (non-interactive)
python main.py --horizon "10 days" --save      # Short-term
python main.py --horizon "3 months" --save    # Medium-term
python main.py --horizon "1 year" --save      # Long-term

# Interactive mode (prompts for ticker and horizon)
python main.py --save

# Disable ML models (statistical models only)
python main.py --no-ml --save

# Adjust number of Monte Carlo scenarios
python main.py --n-scenarios 1000 --save
```

### Options
- `--all-horizons` - **Run forecasts for all horizon types in a single run**
  - Tests: 10 days, 1 month, 3 months, 6 months, 1 year
  - Generates comprehensive comparison table
  - Saves all outputs with horizon-specific filenames
- `--demo` - Run deterministic demo (TSLA, defaults to 10 days, can override with --horizon)
- `--use-sample-data` - Use synthetic data (for offline grading)
- `--save` - Save all plots and outputs to `results/` directory
- `--horizon` - Specify horizon (e.g., "10 days", "3 months", "1 year")
  - Supports: days, weeks, months, years
  - Can combine with --demo to test longer horizons
- `--n-scenarios` - Number of Monte Carlo scenarios (default: 500)
- `--no-ml` - Disable XGBoost (statistical models only)

### Supported Forecast Horizons
The system supports forecasting for any horizon:
- **Short-term**: Days (e.g., "10 days", "30 days")
- **Medium-term**: Weeks or months (e.g., "4 weeks", "3 months")
- **Long-term**: Months or years (e.g., "6 months", "1 year")

Note: For very long horizons (>1 year), forecasts become less precise and the system will warn you.

## Outputs

The project generates the following outputs in the `results/` directory:

1. **Price Forecast Plot** (`forecast_<TICKER>_<TIMESTAMP>.png`)
   - Historical prices
   - Forecasted prices with confidence intervals
   - Monte Carlo percentiles (10th, 50th, 90th)

2. **Volatility Forecast Plot** (`volatility_forecast_<TICKER>_<TIMESTAMP>.png`)
   - Historical rolling volatility
   - GARCH(1,1) forecasted volatility

3. **Model Comparison CSV** (`model_comparison_<TICKER>_<TIMESTAMP>.csv`)
   - RMSE, MAE, MAPE for each model
   - Indicates whether ML models beat the baseline

## Data

- **Primary source**: Yahoo Finance (via `yfinance` library)
- **Automatic caching**: Real data is downloaded once and cached locally in `data/raw/`
- **Reproducibility**: Cached data ensures consistent results across runs
- **Offline capability**: After first download, works offline using cached data
- **Sample data mode**: Use `--use-sample-data` flag for deterministic synthetic data (for testing)
- No API keys required for basic functionality

## Project Structure

```
.
├── main.py                 # Main entry point (required for grading)
├── src/
│   ├── data_loader.py     # Data fetching and preprocessing
│   ├── models.py          # Model implementations (AR(1), ARIMA, XGBoost)
│   └── evaluation.py       # Backtesting, forecasting, and plotting
├── data/
│   └── raw/               # Raw data storage
├── results/               # Generated outputs (plots, CSVs)
├── notebooks/             # Exploratory notebooks
├── environment.yml        # Conda environment file
└── requirements.txt       # pip requirements file
```

## Reproducibility

All random seeds are set to ensure reproducibility:
- NumPy: `np.random.seed(42)`
- XGBoost: `random_state=42`
- Monte Carlo: `np.random.seed(42)`

Run with `--use-sample-data` for fully deterministic results suitable for grading.

## Key Findings

The project explicitly compares all models against the random walk baseline. If machine learning models do not outperform the baseline, this is documented as a valid finding (which is common in financial time series forecasting).

## License

This project is for educational purposes.

## Contact

For questions or issues, please refer to the project documentation or contact the maintainer.
