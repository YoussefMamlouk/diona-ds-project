import os
import re
import sys
import warnings
from typing import Tuple, Dict

# Set environment variables for reproducibility BEFORE importing numerical libraries
os.environ['PYTHONHASHSEED'] = '0'  # For hash-based operations
os.environ['OMP_NUM_THREADS'] = '1'  # For OpenMP (BLAS) - single thread for reproducibility
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # NumExpr
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate framework

import numpy as np
import yfinance as yf
import argparse
import pathlib

# Use package-relative imports. Run the project with `python main.py`.
# Relative imports are the recommended, idiomatic approach for packages.
from src.data_loader import compute_horizon_settings, load_series_for_horizon
from src.evaluation import generate_forecast, plot_forecast, plot_volatility_forecast, save_model_comparison_csv, clean_old_results
from src.eda import generate_eda_report

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


# Hardcoded ticker for reproducibility
TICKER = "TSLA"


def run_single_forecast(
    ticker: str,
    horizon_value: float,
    horizon_unit: str,
    args,
    current_price: float = None
) -> Dict:
    """Run a single forecast for a given horizon and return results.
    
    Returns a dictionary with all forecast artifacts and summary information.
    """
    horizon_settings = compute_horizon_settings(horizon_value, horizon_unit)
    data = load_series_for_horizon(ticker, horizon_settings, 
                                   extra_history_period=args.extra_history, 
                                   use_sample_data=args.use_sample_data,
                                   cache_only=getattr(args, "cache_only", False))

    prices = data.get("prices")
    raw_prices = data.get("raw_prices")
    log_returns = data.get("log_returns")
    exog_df = data.get("exog_df")

    if prices is None or prices.empty:
        return {"error": "No data returned"}

    try:
        artifacts = generate_forecast(ticker, prices, raw_prices, log_returns, exog_df, 
                                     horizon_settings, num_sims=args.n_scenarios, 
                                     use_ml=(not args.no_ml))
    except ValueError as e:
        return {"error": str(e)}

    # Get current price if not provided
    if current_price is None:
        if args.use_sample_data or getattr(args, "cache_only", False):
            current_price = float(prices.iloc[-1])
        else:
            try:
                ticker_hist = yf.Ticker(ticker).history(period="1d")
                if ticker_hist.empty:
                    raise ValueError("empty history")
                current_price = float(ticker_hist["Close"].iloc[-1])
            except Exception:
                current_price = float(prices.iloc[-1])

    artifacts["current_price"] = current_price
    artifacts["horizon_label"] = horizon_settings['label']
    artifacts["horizon_value"] = horizon_value
    artifacts["horizon_unit"] = horizon_unit
    
    return artifacts


def print_forecast_results(ticker: str, artifacts: Dict, horizon_settings: Dict):
    """Print formatted forecast results for a single horizon."""
    sigma_forecast = artifacts.get("sigma_forecast")
    mc_p10 = artifacts.get("mc_p10")
    mc_p50 = artifacts.get("mc_p50")
    mc_p90 = artifacts.get("mc_p90")
    expected_return = artifacts.get("expected_return")
    best_mape = artifacts.get("best_mape")
    signal_quality = artifacts.get("signal_quality")
    all_metrics = artifacts.get("all_metrics", {})
    best_model = artifacts.get("best_model", "unknown")
    forecast_series = artifacts.get("forecast_series")
    current_price = artifacts.get("current_price")

    print("\n" + "="*70)
    print(" " * 15 + f"FORECASTING RESULTS: {horizon_settings['label'].upper()}")
    print("="*70)
    
    print(f"\n{'Asset:':<20} {ticker}")
    print(f"{'Current Price:':<20} ${current_price:.2f}")
    print(f"{'Forecast Horizon:':<20} {horizon_settings['label']}")
    print(f"{'Forecast Steps:':<20} {horizon_settings['steps']} periods")
    
    # Model Comparison Section
    print("\n" + "-"*70)
    print(" " * 20 + "MODEL PERFORMANCE COMPARISON")
    print("-"*70)
    if all_metrics:
        print(f"\n{'Model':<18} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'Status':<15}")
        print("-" * 70)
        baseline_mape = all_metrics.get("random_walk", {}).get("MAPE")
        
        for model_name in ["random_walk", "ar1", "arima", "xgb"]:
            if model_name in all_metrics:
                metrics = all_metrics[model_name]
                if model_name == best_model:
                    status = "✓ BEST MODEL"
                elif model_name == "random_walk":
                    status = "Baseline"
                elif baseline_mape and metrics["MAPE"] < baseline_mape:
                    improvement = ((baseline_mape - metrics["MAPE"]) / baseline_mape) * 100
                    status = f"✓ Beats baseline ({improvement:.1f}% better)"
                else:
                    status = "✗ Below baseline"
                
                model_display = model_name.upper() if model_name != "random_walk" else "Random Walk"
                print(f"{model_display:<18} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['MAPE']:<12.2f}% {status:<15}")
        
        print("-" * 70)
    else:
        print("\n⚠ Model comparison not available (insufficient data)")
    
    # Forecast Summary
    print("\n" + "-"*70)
    print(" " * 25 + "PRICE FORECAST")
    print("-"*70)
    
    if forecast_series is not None and len(forecast_series) > 0:
        final_forecast_price = forecast_series.iloc[-1]
        price_change = final_forecast_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"\n{'Forecasted Price:':<25} ${final_forecast_price:.2f}")
        print(f"{'Expected Change:':<25} ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"{'Expected Return:':<25} {expected_return:+.2f}%")
        
        price_range = mc_p90.iloc[-1] - mc_p10.iloc[-1]
        price_range_pct = (price_range / current_price) * 100
        
        print(f"\n{'Risk Assessment:':<25}")
        print(f"  {'10th Percentile:':<20} ${mc_p10.iloc[-1]:.2f}")
        print(f"  {'Median (50th):':<20} ${mc_p50.iloc[-1]:.2f}")
        print(f"  {'90th Percentile:':<20} ${mc_p90.iloc[-1]:.2f}")
        print(f"  {'Price Range:':<20} ${price_range:.2f} ({price_range_pct:.1f}% of current)")
        
        if price_range_pct > 20:
            risk_level = "HIGH"
        elif price_range_pct > 10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        print(f"  {'Risk Level:':<20} {risk_level}")
    
    # Volatility
    if sigma_forecast is not None and not sigma_forecast.empty:
        avg_vol = float(sigma_forecast.mean())
        print(f"\n{'Average Volatility:':<25} {avg_vol:.2f}% (annualized)")
    
    # Model Quality
    if best_mape is not None:
        print(f"\n{'Best Model:':<25} {best_model.upper().replace('_', ' ')}")
        print(f"{'MAPE:':<25} {best_mape:.2f}%")
        print(f"{'Signal Quality:':<25} {signal_quality.upper()}")


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
    # Make results reproducible by default
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Default (no-args) behavior for reproducibility:
    # Running `python main.py` should behave like:
    #   python main.py --demo --all-horizons --save
    # plus EDA, using ONLY cached data from data/raw/ folder.
    # ------------------------------------------------------------------
    auto_full_run = (len(sys.argv) == 1)
    
    # Hardcoded ticker for reproducibility
    ticker = TICKER

    # Basic arg parsing: allow overriding history period and auto-save
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extra-history", dest="extra_history", help="Override download period (e.g. '2y') to fetch extra history", default=None)
    parser.add_argument("--save", dest="save_plot", help="Save generated plot to results/", action="store_true")
    parser.add_argument("--use-sample-data", dest="use_sample_data", help="Use a deterministic synthetic sample dataset (offline mode)", action="store_true")
    parser.add_argument("--demo", dest="demo", help="Run a deterministic demo (non-interactive). Defaults to 10 days, but can be overridden with --horizon", action="store_true")
    parser.add_argument("--horizon", dest="horizon", help="Provide horizon (e.g. '10 days', '3 months', '1 year') to run non-interactively. Works with --demo to override default 10 days", default=None)
    parser.add_argument("--all-horizons", dest="all_horizons", help="Run forecasts for all horizon types (10 days, 1 month, 3 months, 6 months, 1 year) in a single run", action="store_true")
    parser.add_argument("--n-scenarios", dest="n_scenarios", help="Number of Monte Carlo scenarios (default 500)", type=int, default=500)
    parser.add_argument("--no-ml", dest="no_ml", help="Disable ML models (XGBoost) and run only statistical models", action="store_true")
    parser.add_argument("--eda", dest="run_eda", help="Run exploratory data analysis (EDA) with plots and insights", action="store_true")
    parser.add_argument("--eda-period", dest="eda_period", help="Data period for EDA (e.g., '1y', '2y', '5y', '10y'). Default: '5y'", default="5y")
    parser.add_argument("--cache-only", dest="cache_only", help="Never download data; rely on cached CSV in data/raw/", action="store_true")
    args, _ = parser.parse_known_args()

    if auto_full_run:
        # Reproducible default pipeline
        args.demo = True
        args.all_horizons = True
        args.save_plot = True
        args.run_eda = True
        args.cache_only = True
        # Keep deterministic period; should match committed cache range
        if not args.eda_period:
            args.eda_period = "5y"

    print("\n=============================================\n")
    print("Stock Forecasting & Risk Analysis Tool\n")
    print("=============================================\n")
    if auto_full_run:
        print("Running default full pipeline (EDA + all-horizons forecasts).")
        print("Reproducibility mode: using cached Yahoo Finance CSV only (no downloads).\n")
    else:
        print("This tool will help you analyze the stock of your choice.")
        print("Please provide the following information to proceed:\n")

    # Decide whether this run includes forecasting (vs EDA-only).
    wants_forecasts = auto_full_run or args.all_horizons or (args.horizon is not None) or (not args.run_eda)

    # Always use hardcoded TSLA ticker for reproducibility
    if args.demo:
        # Demo mode: allow horizon override
        if args.horizon:
            # Parse horizon if provided with demo
            try:
                raw = args.horizon
                pattern = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s*$")
                m = pattern.match(raw)
                if not m:
                    raise ValueError("Invalid horizon format")
                hv = float(m.group(1))
                unit = m.group(2).lower()
                unit_map = {"d":"day","day":"day","days":"day","w":"week","week":"week","weeks":"week","m":"month","mo":"month","month":"month","months":"month","y":"year","yr":"year","year":"year","years":"year"}
                if unit not in unit_map:
                    raise ValueError("Unsupported unit")
                horizon_value, horizon_unit = hv, unit_map[unit]
                print(f"Running deterministic demo: {ticker} for {horizon_value} {horizon_unit}(s) (seeded)")
            except Exception:
                print("Invalid --horizon format. Using default 10 days.")
                horizon_value, horizon_unit = 10.0, "day"
                print(f"Running deterministic demo: {ticker} for 10 days (seeded)")
        else:
            # Default demo: 10 days
            horizon_value, horizon_unit = 10.0, "day"
            print(f"Running deterministic demo: {ticker} for 10 days (seeded)")
        # ensure demo saves the plot for inspection
        args.save_plot = True
    else:
        if wants_forecasts:
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
                except Exception:
                    print("Invalid --horizon format. Using default 10 days.")
                    horizon_value, horizon_unit = 10.0, "day"
            else:
                horizon_value, horizon_unit = prompt_horizon()
        else:
            # EDA-only mode: use default horizon
            horizon_value, horizon_unit = 10.0, "day"

    # If we're saving outputs, clean once up-front (avoid deleting EDA outputs later)
    if args.save_plot and (args.run_eda or args.all_horizons):
        clean_old_results(ticker)

    # Run EDA if requested (can be combined with forecasting)
    if args.run_eda:
        try:
            generate_eda_report(ticker, period=args.eda_period, save=True, cache_only=args.cache_only)
            print(f"\n✓ EDA analysis complete for {ticker}")
            print("  EDA plots and statistics saved to results/ directory\n")
        except Exception as e:
            print(f"\n✗ Error running EDA: {e}\n")
            # If the user asked only for EDA, stop here; otherwise continue to forecasting.
            if not wants_forecasts:
                return
        else:
            # Successful EDA-only run: exit unless forecasts were also requested.
            if not wants_forecasts:
                return

    # Handle --all-horizons flag: run forecasts for multiple horizons
    if args.all_horizons:
        # Clean old results before starting all-horizons run
        if args.save_plot and (not args.run_eda):
            clean_old_results(ticker)
        
        # Define standard horizons to test
        all_horizon_configs = [
            (10.0, "day", "10 days"),
            (1.0, "month", "1 month"),
            (3.0, "month", "3 months"),
            (6.0, "month", "6 months"),
            (1.0, "year", "1 year"),
        ]
        
        print("\n" + "="*70)
        print(" " * 15 + "RUNNING FORECASTS FOR ALL HORIZONS")
        print("="*70)
        print(f"\nAsset: {ticker}")
        print(f"Running forecasts for: {', '.join([h[2] for h in all_horizon_configs])}\n")
        
        # Get current price once (will be used for all horizons)
        try:
            if args.use_sample_data or args.cache_only:
                # Get price from first data load
                test_hs = compute_horizon_settings(10.0, "day")
                test_data = load_series_for_horizon(ticker, test_hs, 
                                                   extra_history_period=args.extra_history, 
                                                   use_sample_data=args.use_sample_data,
                                                   cache_only=args.cache_only)
                current_price = float(test_data.get("prices").iloc[-1])
            else:
                ticker_hist = yf.Ticker(ticker).history(period="1d")
                if ticker_hist.empty:
                    raise ValueError("empty history")
                current_price = float(ticker_hist["Close"].iloc[-1])
        except Exception:
            # Fallback: will be set in first forecast
            current_price = None
        
        all_results = []
        all_saved_paths = []
        
        # Run forecast for each horizon
        for horizon_value, horizon_unit, horizon_label in all_horizon_configs:
            print(f"\n{'='*70}")
            print(f"Processing: {horizon_label.upper()}")
            print(f"{'='*70}")
            
            try:
                artifacts = run_single_forecast(ticker, horizon_value, horizon_unit, args, 
                                                current_price)
                
                if "error" in artifacts:
                    print(f"⚠ Error for {horizon_label}: {artifacts['error']}")
                    continue
                
                horizon_settings = compute_horizon_settings(horizon_value, horizon_unit)
                
                # Print results for this horizon
                print_forecast_results(ticker, artifacts, horizon_settings)
                
                # Save outputs with horizon-specific naming
                if args.save_plot:
                    horizon_safe = horizon_label.replace(" ", "_").lower()
                    data = load_series_for_horizon(
                        ticker,
                        horizon_settings,
                        extra_history_period=args.extra_history,
                        use_sample_data=args.use_sample_data,
                        cache_only=args.cache_only,
                    )
                    prices = data.get("prices")
                    raw_prices = data.get("raw_prices")
                    log_returns = data.get("log_returns")
                    
                    # Save price forecast plot with horizon in name
                    saved_path = plot_forecast(ticker, prices, raw_prices, artifacts, save=True, horizon_suffix=horizon_safe)
                    if saved_path:
                        all_saved_paths.append(saved_path)
                    
                    # Save volatility plot
                    sigma_forecast = artifacts.get("sigma_forecast")
                    if sigma_forecast is not None and not sigma_forecast.empty:
                        vol_path = plot_volatility_forecast(
                            ticker,
                            log_returns,
                            sigma_forecast,
                            save=True,
                            horizon_suffix=horizon_safe,
                            raw_prices=raw_prices,
                            sigma_daily_forecast=artifacts.get("sigma_daily_forecast"),
                            sigma_fitted=artifacts.get("sigma_fitted"),
                        )
                        if vol_path:
                            all_saved_paths.append(vol_path)
                    
                    # Save CSV
                    all_metrics = artifacts.get("all_metrics", {})
                    if all_metrics:
                        csv_path = save_model_comparison_csv(all_metrics, ticker, horizon_label, horizon_suffix=horizon_safe)
                        if csv_path:
                            all_saved_paths.append(csv_path)
                
                # Store results for summary
                all_results.append({
                    "horizon": horizon_label,
                    "artifacts": artifacts,
                    "settings": horizon_settings
                })
                
            except Exception as e:
                print(f"⚠ Error processing {horizon_label}: {e}")
                continue
        
        # Print summary comparison across all horizons
        print("\n" + "="*70)
        print(" " * 15 + "SUMMARY: ALL HORIZONS COMPARISON")
        print("="*70)
        
        print(f"\n{'Horizon':<15} {'Best Model':<15} {'MAPE':<10} {'Expected Return':<18} {'Risk Level':<12}")
        print("-" * 70)
        
        for result in all_results:
            horizon = result["horizon"]
            artifacts = result["artifacts"]
            best_model = artifacts.get("best_model", "unknown")
            best_mape = artifacts.get("best_mape")
            expected_return = artifacts.get("expected_return", 0)
            mc_p10 = artifacts.get("mc_p10")
            mc_p50 = artifacts.get("mc_p50")
            mc_p90 = artifacts.get("mc_p90")
            current_price = artifacts.get("current_price", 0)
            
            if mc_p10 is not None and mc_p90 is not None and current_price > 0:
                price_range_pct = ((mc_p90.iloc[-1] - mc_p10.iloc[-1]) / current_price) * 100
                if price_range_pct > 20:
                    risk_level = "HIGH"
                elif price_range_pct > 10:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
            else:
                risk_level = "N/A"
            
            mape_str = f"{best_mape:.2f}%" if best_mape else "N/A"
            return_str = f"{expected_return:+.2f}%" if expected_return else "N/A"
            model_str = best_model.upper().replace("_", " ") if best_model != "unknown" else "N/A"
            
            print(f"{horizon:<15} {model_str:<15} {mape_str:<10} {return_str:<18} {risk_level:<12}")
        
        print("-" * 70)
        
        if all_saved_paths:
            print(f"\n✓ All outputs saved ({len(all_saved_paths)} files)")
            print("  Files include horizon in filename for easy identification")
        
        return
    
    # Single horizon mode (original behavior)
    # derive settings and load prepared series (pass extra history override)
    horizon_settings = compute_horizon_settings(horizon_value, horizon_unit)
    # Notify user when horizon is long and forecasts will be less precise
    if horizon_settings.get("invested_days", 0) > 252:
        print("\nNote: You selected a long investment horizon — forecasts become less precise for longer horizons. Expect wider confidence bands.\n")
    data = load_series_for_horizon(
        ticker,
        horizon_settings,
        extra_history_period=args.extra_history,
        use_sample_data=args.use_sample_data,
        cache_only=args.cache_only,
    )

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
    all_metrics = artifacts.get("all_metrics", {})
    best_model = artifacts.get("best_model", "unknown")
    forecast_series = artifacts.get("forecast_series")
    last_price = artifacts.get("last_price", prices.iloc[-1])

    # Current price - use the last price from the data that was actually used
    # If using sample data, don't fetch real price from yfinance (would cause mismatch)
    if args.use_sample_data or args.cache_only:
        current_price = float(prices.iloc[-1])
    else:
        try:
            ticker_hist = yf.Ticker(ticker).history(period="1d")
            if ticker_hist.empty:
                raise ValueError("empty history")
            current_price = float(ticker_hist["Close"].iloc[-1])
        except Exception:
            current_price = float(prices.iloc[-1])

    print("\n" + "="*70)
    print(" " * 15 + "FORECASTING RESULTS SUMMARY")
    print("="*70)
    
    # Key Information Section
    print(f"\n{'Asset:':<20} {ticker}")
    print(f"{'Current Price:':<20} ${current_price:.2f}")
    print(f"{'Forecast Horizon:':<20} {horizon_settings['label']}")
    print(f"{'Forecast Steps:':<20} {horizon_settings['steps']} periods")
    
    # Model Comparison Section
    print("\n" + "-"*70)
    print(" " * 20 + "MODEL PERFORMANCE COMPARISON")
    print("-"*70)
    if all_metrics:
        print(f"\n{'Model':<18} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'Status':<15}")
        print("-" * 70)
        baseline_mape = all_metrics.get("random_walk", {}).get("MAPE")
        baseline_rmse = all_metrics.get("random_walk", {}).get("RMSE")
        
        for model_name in ["random_walk", "ar1", "arima", "xgb"]:
            if model_name in all_metrics:
                metrics = all_metrics[model_name]
                if model_name == best_model:
                    status = "✓ BEST MODEL"
                elif model_name == "random_walk":
                    status = "Baseline"
                elif baseline_mape and metrics["MAPE"] < baseline_mape:
                    improvement = ((baseline_mape - metrics["MAPE"]) / baseline_mape) * 100
                    status = f"✓ Beats baseline ({improvement:.1f}% better)"
                else:
                    status = "✗ Below baseline"
                
                model_display = model_name.upper() if model_name != "random_walk" else "Random Walk"
                print(f"{model_display:<18} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['MAPE']:<12.2f}% {status:<15}")
        
        print("-" * 70)
        
        # Key Findings
        print("\n📊 KEY FINDINGS:")
        if baseline_mape:
            ml_models = [m for m in all_metrics.keys() if m in ["ar1", "arima", "xgb"]]
            ml_beats_baseline = any(all_metrics[m]["MAPE"] < baseline_mape for m in ml_models if m in all_metrics)
            if ml_beats_baseline:
                best_ml = min([m for m in ml_models if m in all_metrics], 
                             key=lambda m: all_metrics[m]["MAPE"])
                improvement = ((baseline_mape - all_metrics[best_ml]["MAPE"]) / baseline_mape) * 100
                print(f"  ✓ Machine learning models outperform the baseline")
                print(f"  ✓ Best ML model ({best_ml.upper()}) improves MAPE by {improvement:.1f}%")
            else:
                print(f"  ✗ Machine learning models did not beat the random walk baseline")
                print(f"  → This is a valid finding: baseline models are often strong for financial time series")
                print(f"  → The efficient market hypothesis suggests prices follow a random walk")
        
        if best_model == "random_walk":
            print(f"  → Random Walk with Drift is the best performing model")
            print(f"  → Simple mean return forecast outperforms complex models")
        elif best_model in ["ar1", "arima"]:
            print(f"  → Statistical time-series model ({best_model.upper()}) performs best")
            print(f"  → Autocorrelation in returns provides predictive signal")
        elif best_model == "xgb":
            print(f"  → Machine learning model (XGBoost) performs best")
            print(f"  → Non-linear patterns in lagged returns and features are predictive")
    else:
        print("\n⚠ Model comparison not available (insufficient data for train/val/test split)")
        print("  → Need at least 3x forecast horizon + 20 periods for proper evaluation")
    
    # Forecast Summary Section
    print("\n" + "-"*70)
    print(" " * 25 + "PRICE FORECAST")
    print("-"*70)
    
    if forecast_series is not None and len(forecast_series) > 0:
        final_forecast_price = forecast_series.iloc[-1]
        price_change = final_forecast_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"\n{'Forecasted Price:':<25} ${final_forecast_price:.2f}")
        print(f"{'Expected Change:':<25} ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"{'Expected Return:':<25} {expected_return:+.2f}%")
        
        # Risk Assessment
        price_range = mc_p90.iloc[-1] - mc_p10.iloc[-1]
        price_range_pct = (price_range / current_price) * 100
        
        print(f"\n{'Risk Assessment (Monte Carlo):':<25}")
        print(f"  {'10th Percentile:':<20} ${mc_p10.iloc[-1]:.2f} (worst case)")
        print(f"  {'Median (50th):':<20} ${mc_p50.iloc[-1]:.2f} (expected)")
        print(f"  {'90th Percentile:':<20} ${mc_p90.iloc[-1]:.2f} (best case)")
        print(f"  {'Price Range:':<20} ${price_range:.2f} ({price_range_pct:.1f}% of current price)")
        
        if price_range_pct > 20:
            risk_level = "HIGH"
        elif price_range_pct > 10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        print(f"  {'Risk Level:':<20} {risk_level} (based on forecast uncertainty)")
    
    # Volatility Forecast Section
    print("\n" + "-"*70)
    print(" " * 20 + "VOLATILITY FORECAST (GARCH)")
    print("-"*70)
    if sigma_forecast is not None and not sigma_forecast.empty:
        avg_vol = float(sigma_forecast.mean())
        final_vol = float(sigma_forecast.iloc[-1])
        print(f"\n{'Average Forecasted Volatility:':<35} {avg_vol:.2f}% (annualized)")
        print(f"{'Final Period Volatility:':<35} {final_vol:.2f}% (annualized)")
        
        if avg_vol > 30:
            vol_level = "VERY HIGH"
        elif avg_vol > 20:
            vol_level = "HIGH"
        elif avg_vol > 15:
            vol_level = "MODERATE"
        else:
            vol_level = "LOW"
        print(f"{'Volatility Level:':<35} {vol_level}")
        print(f"\n  Note: Volatility forecast based on GARCH(1,1) model")
        print(f"        Higher volatility indicates greater price uncertainty")
    
    # Model Quality Section
    print("\n" + "-"*70)
    print(" " * 25 + "MODEL QUALITY")
    print("-"*70)
    if best_mape is not None:
        print(f"\n{'Best Model:':<25} {best_model.upper().replace('_', ' ')}")
        print(f"{'Backtest MAPE:':<25} {best_mape:.2f}%")
        print(f"{'Signal Quality:':<25} {signal_quality.upper()}")
        
        if signal_quality == "high":
            quality_desc = "Excellent - Model shows strong predictive power"
        elif signal_quality == "medium":
            quality_desc = "Good - Model provides useful forecasts"
        else:
            quality_desc = "Fair - Forecasts have limited accuracy"
        print(f"{'Interpretation:':<25} {quality_desc}")
        
        print(f"\n  → Lower MAPE indicates better forecast accuracy")
        print(f"  → MAPE < 5%: High quality | 5-15%: Medium | >15%: Low")
    else:
        print("\n⚠ Signal quality: UNKNOWN (insufficient data for backtesting)")
        print("  → Need more historical data to evaluate model performance")
    
    print("\n" + "="*70)

    # Clean old results for this ticker before generating new ones
    if args.save_plot:
        clean_old_results(ticker)
    
    # Generate and save all outputs
    saved_paths = []
    
    # 1. Price forecast plot
    # Create horizon suffix for zoom detection
    horizon_label_safe = horizon_settings['label'].replace(" ", "_").lower()
    saved_path = plot_forecast(ticker, prices, raw_prices, artifacts, save=args.save_plot, horizon_suffix=horizon_label_safe)
    if saved_path:
        saved_paths.append(saved_path)
        print(f"\nPrice forecast plot saved to: {saved_path}")
    
    # 2. Volatility forecast plot
    if sigma_forecast is not None and not sigma_forecast.empty:
        vol_path = plot_volatility_forecast(
            ticker,
            log_returns,
            sigma_forecast,
            save=args.save_plot,
            raw_prices=raw_prices,
            sigma_daily_forecast=artifacts.get("sigma_daily_forecast"),
            sigma_fitted=artifacts.get("sigma_fitted"),
        )
        if vol_path:
            saved_paths.append(vol_path)
            print(f"Volatility forecast plot saved to: {vol_path}")
    
    # 3. Model comparison CSV
    if all_metrics:
        csv_path = save_model_comparison_csv(all_metrics, ticker, horizon_settings['label'])
        if csv_path:
            saved_paths.append(csv_path)
            print(f"Model comparison CSV saved to: {csv_path}")
    
    if saved_paths:
        print(f"\nAll outputs saved to results/ directory ({len(saved_paths)} files)")


if __name__ == "__main__":
    main()

