import os
import re
import sys
import warnings
from typing import Tuple

import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import argparse
import pathlib

# Use package-relative imports. Run the package with `python -m src` or use the
# top-level `main.py` launcher. Relative imports are the recommended, idiomatic
# approach for packages and avoid `sys.path` manipulation.
from src.data_loader import compute_horizon_settings, load_series_for_horizon
from src.evaluation import generate_forecast, plot_forecast

warnings.filterwarnings("ignore")
try:
    # Setting the timezone cache location must be done before the cache is
    # initialized. Tests or other imports may have already created the
    # cache, which raises an AssertionError; guard against that so importing
    # this module is side-effect free for tests.
    yf.set_tz_cache_location("~/.cache/yfinance")
except AssertionError:
    # Cache already initialized; ignore to keep import-time idempotency.
    pass
except Exception as e:
    warnings.warn(f"Could not set yfinance tz cache location: {e}")


def fetch_news_for_ticker(ticker: str, api_key):
    """Fetch recent news articles for `ticker` using NewsAPI.org.

    Returns a list of article dicts (may be empty). Caller should handle
    network errors gracefully.
    """
    if not api_key:
        return []
    try:
        import requests
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []


def prompt_ticker() -> str:
    while True:
        print(" You can find the ticker symbol of your chosen stock at https://finance.yahoo.com/lookup\n")
        ticker = input(" Enter your ticker symbol (example: TSLA): ").upper().strip()
        if ticker:
            return ticker
        print(" Invalid input. Please enter a valid ticker symbol.\n")


def prompt_horizon() -> Tuple[float, str]:
    """Parse horizon like '6 days', '3 months', '1 year' and return (value, normalized_unit)."""
    unit_map = {
        "d": "day",
        "day": "day",
        "days": "day",
        "w": "week",
        "week": "week",
        "weeks": "week",
        "m": "month",
        "mo": "month",
        "month": "month",
        "months": "month",
        "y": "year",
        "yr": "year",
        "year": "year",
        "years": "year",
    }
    pattern = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s*$")
    while True:
        raw = input(" Enter your investment horizon (e.g., '6 days', '3 months', '1 year'): ").lower()
        match = pattern.match(raw)
        if not match:
            print(" Invalid format. Please type a number followed by a unit (days/weeks/months/years).")
            continue
        value = float(match.group(1))
        unit = unit_map.get(match.group(2))
        if unit is None:
            print(" Unsupported unit. Please use days, weeks, months, or years.")
            continue
        return value, unit

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    fred_api_key = os.getenv("FRED_API_KEY", "dde92ad42f9e89f8f7d889a4b79f2efb")

    # Make results reproducible by default
    np.random.seed(42)

    if not api_key or api_key.strip() == "":
        print("\n No API key detected in .env.")
        print("News features will be disabled unless you manually add a free API key.")
        print("To enable news features, create a free key at:")
        print("   https://newsapi.org/")
        print("Then add it in your local .env file as:")
        print("   API_KEY=your_key_here\n")

    print("\n=============================================\n")
    print("Stock Forecasting & Risk Analysis Tool\n")
    print("=============================================\n")
    print("This tool will help you analyze the stock of your choice.")
    print("Please provide the following information to proceed:\n")

    # Basic arg parsing: allow overriding history period and auto-save
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-history", dest="extra_history", help="Override download period (e.g. '2y') to fetch extra history", default=None)
    parser.add_argument("--save", dest="save_plot", help="Save generated plot to results/", action="store_true")
    parser.add_argument("--use-sample-data", dest="use_sample_data", help="Use a deterministic synthetic sample dataset (offline mode)", action="store_true")
    parser.add_argument("--demo", dest="demo", help="Run a deterministic demo (non-interactive)", action="store_true")
    parser.add_argument("--horizon", dest="horizon", help="Provide horizon (e.g. '10 days' or '3 months') to run non-interactively", default=None)
    parser.add_argument("--n-scenarios", dest="n_scenarios", help="Number of Monte Carlo scenarios (default 500)", type=int, default=500)
    parser.add_argument("--no-ml", dest="no_ml", help="Disable ML models (XGBoost) and run only statistical models", action="store_true")
    args, _ = parser.parse_known_args()

    if args.demo:
        print("Running deterministic demo: TSLA for 10 days (seeded)")
        ticker = "TSLA"
        horizon_value, horizon_unit = 10.0, "day"
        # ensure demo saves the plot for inspection
        args.save_plot = True
    else:
        if args.horizon:
            # parse horizon argument like '10 days'
            try:
                raw = args.horizon
                pattern = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s*$")
                m = pattern.match(raw)
                if not m:
                    raise ValueError("Invalid horizon format")
                hv = float(m.group(1))
                unit = m.group(2).lower()
                # map units similar to prompt_horizon
                unit_map = {"d":"day","day":"day","days":"day","w":"week","week":"week","weeks":"week","m":"month","mo":"month","month":"month","months":"month","y":"year","yr":"year","year":"year","years":"year"}
                if unit not in unit_map:
                    raise ValueError("Unsupported unit")
                horizon_value, horizon_unit = hv, unit_map[unit]
                ticker = prompt_ticker() if not args.demo and not args.use_sample_data else "TSLA"
            except Exception:
                print("Invalid --horizon format. Falling back to interactive prompt.")
                ticker = prompt_ticker()
                horizon_value, horizon_unit = prompt_horizon()
        else:
            ticker = prompt_ticker()
            horizon_value, horizon_unit = prompt_horizon()

    # derive settings and load prepared series (pass extra history override)
    horizon_settings = compute_horizon_settings(horizon_value, horizon_unit)
    # Notify user when horizon is long and forecasts will be less precise
    if horizon_settings.get("invested_days", 0) > 252:
        print("\nNote: You selected a long investment horizon — forecasts become less precise for longer horizons. Expect wider confidence bands.\n")
    data = load_series_for_horizon(ticker, horizon_settings, fred_api_key, extra_history_period=args.extra_history, use_sample_data=args.use_sample_data)

    prices = data.get("prices")
    raw_prices = data.get("raw_prices")
    log_returns = data.get("log_returns")
    exog_df = data.get("exog_df")

    if prices is None or prices.empty:
        print("No data returned for the selected ticker and period. Please try another ticker or shorter horizon.")
        sys.exit(1)

    try:
        artifacts = generate_forecast(ticker, prices, raw_prices, log_returns, exog_df, horizon_settings, num_sims=args.n_scenarios, use_ml=(not args.no_ml))
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # Print summary results
    sigma_forecast = artifacts.get("sigma_forecast")
    mc_p10 = artifacts.get("mc_p10")
    mc_p50 = artifacts.get("mc_p50")
    mc_p90 = artifacts.get("mc_p90")
    expected_return = artifacts.get("expected_return")
    best_mape = artifacts.get("best_mape")
    signal_quality = artifacts.get("signal_quality")

    print("\nForecasted Volatility (Standard Deviation):\n", sigma_forecast)
    print("\nMonte Carlo scenarios (log-return AR/GARCH/XGB):")
    print(f" Final price 10th percentile: {mc_p10.iloc[-1]:.2f}")
    print(f" Final price median         : {mc_p50.iloc[-1]:.2f}")
    print(f" Final price 90th percentile: {mc_p90.iloc[-1]:.2f}")
    print(f"Expected return over {horizon_settings['label']}: {expected_return:.2f}%")
    if best_mape is not None:
        print(f"Signal quality: {signal_quality.upper()} (backtest MAPE {best_mape:.2f}%)")
    else:
        print("Signal quality: UNKNOWN (no backtest window available)")

    # Current price
    try:
        ticker_hist = yf.Ticker(ticker).history(period="1d")
        if ticker_hist.empty:
            raise ValueError("empty history")
        current_price = float(ticker_hist["Close"].iloc[-1])
        print(f"Current price of {ticker}: ${current_price:.2f}")
    except Exception:
        current_price = float(prices.iloc[-1])
        print(f"Current price of {ticker}: ${current_price:.2f} (fallback from downloaded data)")

    # News (delegated to local function)
    articles = fetch_news_for_ticker(ticker, api_key)
    if not api_key or api_key.strip() == "":
        print("\n News feature disabled (no API key provided).\n")
    elif not articles:
        print("\n No recent news articles found or failed to fetch news.\n")
    else:
        print(f"\nRecent news articles about {ticker}:\n")
        for article in articles[:5]:
            print(f"Title: {article.get('title')}")
            print(f"Description: {article.get('description')}\n")

    # Plot (optionally save)
    saved_path = plot_forecast(ticker, prices, raw_prices, artifacts, save=args.save_plot)
    if saved_path:
        print(f"Plot saved to: {saved_path}")


if __name__ == "__main__":
    main()

