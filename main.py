import os
import warnings
from typing import Dict

# Set environment variables for reproducibility BEFORE importing numerical libraries
os.environ['PYTHONHASHSEED'] = '0'  # For hash-based operations
os.environ['OMP_NUM_THREADS'] = '1'  # For OpenMP (BLAS) - single thread for reproducibility
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # NumExpr
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate framework

import numpy as np
import argparse
import pathlib

# Use package-relative imports. Run the project with `python main.py`.
# Relative imports are the recommended, idiomatic approach for packages.
from src.data_loader import compute_horizon_settings, load_series_for_horizon
from src.evaluation import (
    generate_forecast,
    plot_forecast,
    plot_volatility_forecast,
    save_model_comparison_csv,
    clean_old_results,
)
from src.eda import generate_eda_report

warnings.filterwarnings("ignore")


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
    data = load_series_for_horizon(ticker, horizon_settings, cache_only=args.cache_only)

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

    # Get current price if not provided.
    # Avoid extra Yahoo Finance calls: use the last price from the loaded dataset.
    if current_price is None:
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
    best_mape = artifacts.get("best_mape")  # test MAPE (final)
    signal_quality = artifacts.get("signal_quality")
    all_metrics = artifacts.get("all_metrics", {})  # test metrics (final)
    validation_metrics = artifacts.get("validation_metrics", {})  # selection metrics (validation)
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
    def _print_metrics_table(metrics_dict: Dict[str, Dict[str, float]], title: str, selected_model: str):
        if not metrics_dict:
            print(f"\n⚠ {title}: not available (insufficient data)")
            return
        print(f"\n{title}")
        print(f"{'Model':<18} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'Status':<22}")
        print("-" * 70)
        baseline_rmse = metrics_dict.get("random_walk", {}).get("RMSE")
        order = ["random_walk", "ridge", "lasso", "elasticnet", "ar1", "arima", "xgb"]
        for model_name in order:
            if model_name not in metrics_dict:
                continue
            metrics = metrics_dict[model_name]
            if model_name == selected_model:
                status = "✓ SELECTED (val best)"
            elif model_name == "random_walk":
                status = "Baseline"
            elif baseline_rmse and metrics["RMSE"] < baseline_rmse:
                improvement = ((baseline_rmse - metrics["RMSE"]) / baseline_rmse) * 100
                status = f"✓ Beats baseline ({improvement:.1f}% rmse)"
            else:
                status = "✗ Below baseline"
            model_display = model_name.upper() if model_name != "random_walk" else "Random Walk"
            print(
                f"{model_display:<18} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['MAPE']:<12.2f}% {status:<22}"
            )
        print("-" * 70)

    _print_metrics_table(validation_metrics, "Validation (used for selection)", best_model)
    _print_metrics_table(all_metrics, "Test (final; refit on train+val)", best_model)
    
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


def main():
    # Make results reproducible by default
    np.random.seed(42)

    # Hardcoded ticker for reproducibility
    ticker = TICKER

    # Basic arg parsing: keep only essential controls
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save", dest="save_plot", help="Save generated plot to results/", action="store_true")
    parser.add_argument("--n-scenarios", dest="n_scenarios", help="Number of Monte Carlo scenarios (default 500)", type=int, default=500)
    parser.add_argument("--no-ml", dest="no_ml", help="Disable ML models (XGBoost) and run only statistical models", action="store_true")
    parser.add_argument("--cache-only", dest="cache_only", help="Use cached data only (no downloads)", action="store_true")
    args, _ = parser.parse_known_args()

    standard_horizons = [
        (10.0, "day", "10 days"),
        (1.0, "month", "1 month"),
        (3.0, "month", "3 months"),
        (6.0, "month", "6 months"),
        (1.0, "year", "1 year"),
    ]
    horizon_labels = ", ".join([h[2] for h in standard_horizons])

    print("Running default full pipeline (EDA + all-horizons forecasts).")
    if args.cache_only:
        print("Reproducibility mode: using cached Yahoo Finance CSV only (no downloads).")
    else:
        print("Cache-first mode: uses cached CSV when available and downloads once if missing.")
    print(f"Forecast horizons: {horizon_labels}\n")

    if args.save_plot:
        clean_old_results(ticker)

    try:
        generate_eda_report(ticker, save=args.save_plot, cache_only=args.cache_only)
        print(f"\n✓ EDA analysis complete for {ticker}")
        if args.save_plot:
            print("  EDA plots and statistics saved to results/ directory\n")
    except Exception as e:
        print(f"\n✗ Error running EDA: {e}\n")

    print("\n" + "="*70)
    print(" " * 15 + "RUNNING FORECASTS FOR ALL HORIZONS")
    print("="*70)
    print(f"\nAsset: {ticker}")
    print(f"Running forecasts for: {', '.join([h[2] for h in standard_horizons])}\n")

    try:
        test_hs = compute_horizon_settings(10.0, "day")
        test_data = load_series_for_horizon(
            ticker,
            test_hs,
            cache_only=args.cache_only,
        )
        current_price = float(test_data.get("prices").iloc[-1])
    except Exception:
        current_price = None

    all_results = []
    all_saved_paths = []

    for horizon_value, horizon_unit, horizon_label in standard_horizons:
        print(f"\n{'='*70}")
        print(f"Processing: {horizon_label.upper()}")
        print(f"{'='*70}")

        try:
            artifacts = run_single_forecast(ticker, horizon_value, horizon_unit, args, current_price)

            if "error" in artifacts:
                print(f"⚠ Error for {horizon_label}: {artifacts['error']}")
                continue

            horizon_settings = compute_horizon_settings(horizon_value, horizon_unit)
            print_forecast_results(ticker, artifacts, horizon_settings)

            if args.save_plot:
                horizon_safe = horizon_label.replace(" ", "_").lower()
                data = load_series_for_horizon(
                    ticker,
                    horizon_settings,
                    cache_only=args.cache_only,
                )
                prices = data.get("prices")
                raw_prices = data.get("raw_prices")
                log_returns = data.get("log_returns")

                saved_path = plot_forecast(
                    ticker,
                    prices,
                    raw_prices,
                    artifacts,
                    save=True,
                    horizon_suffix=horizon_safe,
                )
                if saved_path:
                    all_saved_paths.append(saved_path)

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
                        garch_model_label=artifacts.get("garch_model_label"),
                    )
                    if vol_path:
                        all_saved_paths.append(vol_path)

                test_metrics = artifacts.get("all_metrics", {})
                val_metrics = artifacts.get("validation_metrics", {})
                if val_metrics:
                    csv_path = save_model_comparison_csv(
                        val_metrics, ticker, horizon_label, horizon_suffix=horizon_safe, dataset_label="validation"
                    )
                    if csv_path:
                        all_saved_paths.append(csv_path)
                if test_metrics:
                    csv_path = save_model_comparison_csv(
                        test_metrics, ticker, horizon_label, horizon_suffix=horizon_safe, dataset_label="test"
                    )
                    if csv_path:
                        all_saved_paths.append(csv_path)

            all_results.append({
                "horizon": horizon_label,
                "artifacts": artifacts,
                "settings": horizon_settings,
            })

        except Exception as e:
            print(f"⚠ Error processing {horizon_label}: {e}")
            continue

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

    print("\n" + "-"*70)
    print(" " * 15 + "VOLATILITY BACKTEST SUMMARY")
    print("-"*70)
    print(f"{'Horizon':<15} {'Model':<12} {'RMSE':<12} {'MAE':<12} {'QLIKE':<12} {'Window':<10} {'Points':<10}")
    print("-" * 70)
    for result in all_results:
        horizon = result["horizon"]
        artifacts = result["artifacts"]
        vol_backtest = artifacts.get("vol_backtest", {})
        metrics = vol_backtest.get("metrics", {}) if isinstance(vol_backtest, dict) else {}
        window = vol_backtest.get("rolling_window") if isinstance(vol_backtest, dict) else None
        n_obs = vol_backtest.get("points") if isinstance(vol_backtest, dict) else None
        window_str = str(int(window)) if window else "N/A"
        n_obs_str = str(int(n_obs)) if n_obs else "N/A"

        def _row(label: str, m: Dict[str, float], horizon_label: str):
            rmse = m.get("RMSE")
            mae = m.get("MAE")
            qlike = m.get("QLIKE")
            rmse_str = f"{rmse:.4f}%" if rmse is not None else "N/A"
            mae_str = f"{mae:.4f}%" if mae is not None else "N/A"
            qlike_str = f"{qlike:.4f}" if qlike is not None else "N/A"
            print(f"{horizon_label:<15} {label:<12} {rmse_str:<12} {mae_str:<12} {qlike_str:<12} {window_str:<10} {n_obs_str:<10}")

        garch_m = metrics.get("garch", {})
        if garch_m:
            _row("GARCH", garch_m, horizon)
        ewma_m = metrics.get("baseline_ewma", {})
        if ewma_m:
            _row("BASELINE", ewma_m, "")
    print("-" * 70)

    if all_saved_paths:
        print(f"\n✓ All outputs saved ({len(all_saved_paths)} files)")
        print("  Files include horizon in filename for easy identification")


if __name__ == "__main__":
    main()
