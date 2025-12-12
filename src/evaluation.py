"""
Evaluation module: model training, backtesting, forecasting, and Monte Carlo simulation.

This module coordinates the forecasting pipeline by:
- Training ARIMA/XGBoost models on historical data
- Running backtests to select the best performer
- Generating forecasts with uncertainty quantification via Monte Carlo
- Adapting to long vs short horizon characteristics
"""
from .lib.model_utils import run_backtest
from .models import forecast_from_arima, forecast_with_xgb
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .plotting import plot_forecast_save
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


def evaluate_backtest(log_returns, exog_df, forecast_periods, model_type, prices):
    """Run the existing backtest and return (best_model_name, best_mape, signal_quality)."""
    return run_backtest(log_returns, exog_df, forecast_periods, model_type, prices)


def generate_forecast(
    ticker: str,
    prices: pd.Series,
    raw_prices: pd.Series,
    log_returns: pd.Series,
    exog_df: pd.DataFrame,
    horizon_settings: Dict[str, object],
    num_sims: int = 500,
    use_ml: bool = True
) -> Dict[str, object]:
    """Generate price forecasts with uncertainty quantification.
    
    Pipeline:
    1. Train ARIMA and XGBoost models
    2. Backtest to select best performer
    3. Generate point forecast from winning model
    4. Estimate volatility using GARCH
    5. Run Monte Carlo simulation for probabilistic forecasts
    
    For long horizons (>20 steps), uses historical volatility and drift instead of
    ARIMA mean-reversion to avoid unrealistic flat forecasts.
    
    Args:
        ticker: Stock symbol
        prices: Historical price series (log-transformed)
        raw_prices: Original price series
        log_returns: Log returns series
        exog_df: Exogenous variables (VIX, rates, etc.)
        horizon_settings: Dict with 'steps', 'interval', 'mode', 'label'
        num_sims: Number of Monte Carlo scenarios
        use_ml: Whether to use XGBoost (if False, only statistical models)
        
    Returns:
        Dict containing forecast_series, confidence bounds, MC percentiles,
        expected return, best model info, and diagnostics
    """
    # Step 1: Fit ARIMA model (with or without exogenous variables)
    has_exog = not exog_df.empty and exog_df.shape[1] > 0

    if horizon_settings["mode"] == "D":
        # Short daily horizons: use fixed AR(1) model
        if not exog_df.empty:
            arima_full = ARIMA(log_returns, order=(1, 0, 0), exog=exog_df).fit()
        else:
            arima_full = ARIMA(log_returns, order=(1, 0, 0)).fit()
        model_type = "arima_fixed"
    else:
        # Longer horizons: auto-select ARIMA order
        arima_full = auto_arima(
            log_returns,
            exogenous=exog_df if not exog_df.empty else None,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            trace=False,
        )
        model_type = "auto_arima"

    drift_full = float(log_returns.mean())
    forecast_steps = horizon_settings["steps"]

    # Step 2: Determine forecast date range and frequency
    if horizon_settings["interval"] == "1d":
        start = prices.index[-1] + pd.offsets.BDay()
        freq = "B"
    elif horizon_settings["interval"] == "1wk":
        start = prices.index[-1] + pd.offsets.Week(weekday=4)
        freq = "W-FRI"
    elif horizon_settings["interval"] == "1mo":
        start = prices.index[-1] + pd.offsets.MonthBegin(1)
        freq = "MS"
    else:
        raise ValueError("Unsupported interval period.")

    # Step 3: Run backtest to select best model (ARIMA vs XGBoost vs Drift)
    best_model_name = "arima"
    best_mape = None
    signal_quality = "unknown"
    
    if len(prices) > forecast_steps * 2 and len(log_returns) > forecast_steps + 5:
        best_model_name, best_mape, signal_quality = run_backtest(
            log_returns, exog_df, forecast_steps, model_type, prices
        )

    # Step 4: Generate forecast returns using the selected best model
    if best_model_name == "arima":
        exog_future = None
        if not exog_df.empty:
            last = exog_df.iloc[-1].values
            exog_future = np.vstack([last] * forecast_steps)
        forecast_returns = forecast_from_arima(arima_full, model_type, forecast_steps, exog_future)
    elif best_model_name == "drift":
        forecast_returns = np.full(forecast_steps, drift_full)
    elif best_model_name == "xgb" and use_ml:
        try:
            from .models import train_xgb_cv
            xgb_model = train_xgb_cv(log_returns, exog_df if has_exog else None)
            if xgb_model is not None:
                forecast_returns = forecast_with_xgb(log_returns, exog_df if has_exog else None, forecast_steps, model=xgb_model)
            else:
                forecast_returns = forecast_from_arima(arima_full, model_type, forecast_steps)
        except Exception:
            forecast_returns = forecast_from_arima(arima_full, model_type, forecast_steps)
    else:
        forecast_returns = np.full(forecast_steps, drift_full)

    # Step 5: Adjust forecast for long horizons
    # ARIMA mean-reverts to zero, which creates unrealistic flat forecasts for long periods
    # For horizons >20 steps, use historical drift as baseline (random walk with drift)
    if best_model_name == "arima" and forecast_steps > 20:
        historical_drift = float(log_returns.mean())
        historical_vol = float(log_returns.std())
        # Generate ONE random path with drift + volatility for the center forecast
        np.random.seed(999)  # Different seed from MC
        random_shocks = np.random.normal(0, historical_vol, forecast_steps)
        adjusted_returns = historical_drift + random_shocks
    else:
        adjusted_returns = forecast_returns
    
    # Step 6: Convert log-returns to prices
    cum_returns = np.cumsum(adjusted_returns)
    last_price = float(prices.iloc[-1])
    forecast_prices = last_price * np.exp(cum_returns)
    future_dates = pd.date_range(start=start, periods=forecast_steps, freq=freq)
    forecast_series = pd.Series(forecast_prices, index=future_dates)

    # Step 7: Estimate future volatility using GARCH(1,1)
    returns = 100 * log_returns.dropna()
    if returns.empty:
        raise ValueError("Insufficient return data to estimate volatility.")
    
    try:
        from arch import arch_model
    except Exception as e:
        raise ImportError(
            "The 'arch' package is required to estimate forward volatility but failed to import.\n"
            "Install it with: python -m pip install arch==5.2.0\n"
            "If installation fails on Windows, try: conda install -c conda-forge arch\n"
            "Original import error: %s" % e
        ) from e

    garch = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_steps)
    sigma_forecast = pd.Series(garch_forecast.variance.values[-1, :] ** 0.5, index=future_dates)
    
    # Adjust volatility based on horizon (amplify for long-term uncertainty)
    if forecast_steps > 20:
        sigma_soft = sigma_forecast * 1.5
    else:
        sigma_soft = sigma_forecast * 0.8

    # Step 8: Monte Carlo simulation for probabilistic forecasts
    num_sims = int(num_sims)
    np.random.seed(42)  # For reproducibility
    
    # Choose volatility source: historical (long horizon) or GARCH (short horizon)
    if forecast_steps > 20:
        historical_vol = float(log_returns.std())
        # Double historical vol for long-term to reflect increasing uncertainty
        sigma_for_mc = np.full(forecast_steps, historical_vol * 2.0)
    else:
        sigma_for_mc = sigma_soft.values / 100.0
    
    # Generate random shocks and simulate price paths
    shocks = np.random.normal(size=(num_sims, forecast_steps)) * sigma_for_mc
    paths = last_price * np.exp(np.cumsum(adjusted_returns + shocks, axis=1))
    
    # Extract percentiles for confidence bands
    narrow_percentiles = np.percentile(paths, [10, 50, 90], axis=0)
    mc_p10 = pd.Series(narrow_percentiles[0], index=future_dates)
    mc_p50 = pd.Series(narrow_percentiles[1], index=future_dates)
    mc_p90 = pd.Series(narrow_percentiles[2], index=future_dates)

    # Step 9: For long ARIMA forecasts, confidence bounds are MC percentiles
    if best_model_name == "arima" and forecast_steps > 20:
        # For long horizons, the forecast_series already has volatility from adjusted_returns
        # Use MC percentiles as confidence bounds
        hybrid_forecast_upper = mc_p90.copy()
        hybrid_forecast_lower = mc_p10.copy()
    else:
        # Short horizons: use GARCH-based bounds
        impact_vol = last_price * (sigma_soft / 100)
        hybrid_forecast_upper = forecast_series + 1.96 * impact_vol
        hybrid_forecast_lower = forecast_series - 1.96 * impact_vol
    
    expected_return = ((mc_p50.iloc[-1] - last_price) / last_price) * 100

    return {
        "forecast_series": forecast_series,
        "hybrid_upper": hybrid_forecast_upper,
        "hybrid_lower": hybrid_forecast_lower,
        "mc_p10": mc_p10,
        "mc_p50": mc_p50,
        "mc_p90": mc_p90,
        "expected_return": expected_return,
        "best_model": best_model_name,
        "best_mape": best_mape,
        "signal_quality": signal_quality,
        "sigma_forecast": sigma_forecast,
        "last_price": last_price,
        "forecast_returns": forecast_returns,
    }


def plot_forecast(ticker: str, prices: pd.Series, raw_prices: pd.Series, forecast_artifacts: Dict[str, object], save: bool = False) -> Optional[str]:
    """Delegate plotting to `src.plotting.plot_forecast_save` which can save the figure.

    Returns the saved file path if `save=True`, else None.
    """
    return plot_forecast_save(ticker, prices, raw_prices, forecast_artifacts, save=save)
