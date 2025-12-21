"""
Evaluation module: model training, backtesting, forecasting, and Monte Carlo simulation.

This module coordinates the forecasting pipeline by:
- Training ARIMA/XGBoost models on historical data
- Running backtests to select the best performer
- Generating forecasts with uncertainty quantification via Monte Carlo
- Adapting to long vs short horizon characteristics
"""
from .models import forecast_from_arima, forecast_with_xgb, train_ar1, forecast_ar1
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from pmdarima import auto_arima


def run_backtest(
    log_returns: pd.Series,
    exog_df: pd.DataFrame,
    forecast_periods: int,
    model_type: str,
    prices: pd.Series,
    use_ml: bool = True,
) -> Tuple[str, Optional[float], str, Dict[str, Dict[str, float]]]:
    """Run backtests on multiple models with proper train/validation/test split.
    
    Sets random seed for reproducibility before model training.
    
    Uses time-series aware splitting:
    - Train set: initial portion for model training
    - Validation set: middle portion for model selection
    - Test set: final portion for final evaluation only
    
    Args:
        log_returns: Log returns series
        exog_df: Exogenous features DataFrame
        forecast_periods: Number of periods to forecast (used for test set size)
        model_type: Type of model ('arima_fixed' or 'auto_arima')
        prices: Price series for computing actual values
        use_ml: Whether to evaluate XGBoost model
    
    Returns:
        Tuple of (best_model_name, best_mape, signal_quality, all_metrics_dict)
        where all_metrics_dict contains RMSE, MAE, MAPE for each model
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Need at least 3x forecast_periods for train/val/test split
    min_required = forecast_periods * 3 + 20
    if len(log_returns) < min_required:
        return "random_walk", None, "unknown", {}
    
    # Time-series aware split: train / validation / test
    # Test set: last forecast_periods
    # Validation set: forecast_periods before test
    # Train set: everything before validation
    test_size = forecast_periods
    val_size = forecast_periods
    train_size = len(log_returns) - val_size - test_size
    
    if train_size < 20:
        return "random_walk", None, "unknown", {}
    
    # Split data chronologically (no shuffling - time series!)
    train_returns = log_returns.iloc[:train_size]
    val_returns = log_returns.iloc[train_size:train_size + val_size]
    test_returns = log_returns.iloc[train_size + val_size:]
    
    train_prices = prices.iloc[:train_size]
    val_prices = prices.iloc[train_size:train_size + val_size]
    test_prices = prices.iloc[train_size + val_size:]
    
    has_exog = not exog_df.empty and exog_df.shape[1] > 0
    train_exog = exog_df.iloc[:train_size] if has_exog else None
    val_exog = exog_df.iloc[train_size:train_size + val_size] if has_exog else None
    test_exog = exog_df.iloc[train_size + val_size:] if has_exog else None
    
    # Dictionary to store all metrics for all models
    all_metrics = {}
    
    # Helper function to calculate all metrics
    def calculate_metrics(forecast_prices: np.ndarray, actual_prices: np.ndarray) -> Dict[str, float]:
        """Calculate RMSE, MAE, and MAPE."""
        rmse = float(np.sqrt(mean_squared_error(actual_prices, forecast_prices)))
        mae = float(mean_absolute_error(actual_prices, forecast_prices))
        mape = float(mean_absolute_percentage_error(actual_prices, forecast_prices) * 100)
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    
    # 1. Random Walk with Drift (baseline)
    try:
        drift = float(train_returns.mean())
        forecast_returns = np.full(forecast_periods, drift)
        last_price = val_prices.iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = test_prices.iloc[:forecast_periods].values
        all_metrics["random_walk"] = calculate_metrics(forecast_prices, actual_prices)
    except Exception:
        pass
    
    # 2. AR(1) model
    # Note: AR(1) models mean-revert quickly and may perform poorly on long horizons
    try:
        # Ensure seed is set before AR(1) training
        np.random.seed(42)
        ar1_model = train_ar1(train_returns, train_exog)
        if val_exog is not None:
            val_exog_array = np.vstack([val_exog.iloc[-1].values] * forecast_periods)
        else:
            val_exog_array = None
        forecast_returns = forecast_ar1(ar1_model, forecast_periods, val_exog_array)
        last_price = val_prices.iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = test_prices.iloc[:forecast_periods].values
        metrics = calculate_metrics(forecast_prices, actual_prices)
        # Only include if MAPE is reasonable (<100% to avoid extreme failures)
        if metrics["MAPE"] < 100:
            all_metrics["ar1"] = metrics
    except Exception:
        pass
    
    # 3. ARIMA model
    try:
        # Ensure seed is set before ARIMA fitting
        np.random.seed(42)
        if model_type == "arima_fixed":
            if has_exog and train_exog is not None:
                arima_model = ARIMA(train_returns, order=(1, 0, 0), exog=train_exog).fit()
                if val_exog is not None:
                    val_exog_array = np.vstack([val_exog.iloc[-1].values] * forecast_periods)
                    forecast_returns = arima_model.forecast(steps=forecast_periods, exog=val_exog_array).values
                else:
                    forecast_returns = arima_model.forecast(steps=forecast_periods).values
            else:
                arima_model = ARIMA(train_returns, order=(1, 0, 0)).fit()
                forecast_returns = arima_model.forecast(steps=forecast_periods).values
        else:
            # Auto ARIMA
            # Set random seed before auto_arima for reproducibility
            np.random.seed(42)
            arima_model = auto_arima(
                train_returns,
                exogenous=train_exog if has_exog else None,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False,
                random_state=42,  # For reproducibility
                n_jobs=1,  # Single thread for reproducibility
            )
            if val_exog is not None:
                val_exog_array = np.vstack([val_exog.iloc[-1].values] * forecast_periods)
                forecast_returns = forecast_from_arima(arima_model, "auto_arima", forecast_periods, val_exog_array)
            else:
                forecast_returns = forecast_from_arima(arima_model, "auto_arima", forecast_periods)
        
        last_price = val_prices.iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = test_prices.iloc[:forecast_periods].values
        all_metrics["arima"] = calculate_metrics(forecast_prices, actual_prices)
    except Exception:
        pass
    
    # 4. XGBoost (if enabled)
    # Note: XGBoost iterative forecasting accumulates errors on long horizons
    if use_ml:
        try:
            from .models import forecast_with_xgb, train_xgb_cv
            
            xgb_model = train_xgb_cv(train_returns, train_exog)
            if xgb_model is not None:
                # Use validation data for forecasting context
                combined_returns = pd.concat([train_returns, val_returns])
                combined_exog = pd.concat([train_exog, val_exog]) if train_exog is not None else None
                forecast_returns = forecast_with_xgb(combined_returns, combined_exog, forecast_periods, model=xgb_model)
                if forecast_returns is not None:
                    last_price = val_prices.iloc[-1]
                    forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
                    actual_prices = test_prices.iloc[:forecast_periods].values
                    metrics = calculate_metrics(forecast_prices, actual_prices)
                    # Only include if MAPE is reasonable (<100% to avoid extreme failures on long horizons)
                    if metrics["MAPE"] < 100:
                        all_metrics["xgb"] = metrics
        except Exception:
            pass
    
    if not all_metrics:
        return "random_walk", None, "unknown", {}
    
    # Find best model based on MAPE (on validation set, but we use test for final comparison)
    # For model selection, we'd use validation set, but for reporting we use test set
    best_model = min(all_metrics, key=lambda k: all_metrics[k]["MAPE"])
    best_mape = all_metrics[best_model]["MAPE"]
    
    # Determine signal quality
    if best_mape < 5:
        signal_quality = "high"
    elif best_mape < 15:
        signal_quality = "medium"
    else:
        signal_quality = "low"
    
    return best_model, best_mape, signal_quality, all_metrics


def evaluate_backtest(log_returns, exog_df, forecast_periods, model_type, prices, use_ml=True):
    """Run the existing backtest and return (best_model_name, best_mape, signal_quality, all_metrics)."""
    return run_backtest(log_returns, exog_df, forecast_periods, model_type, prices, use_ml=use_ml)


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
    # Set random seed for reproducibility (models may use random initialization)
    np.random.seed(42)
    
    # Step 1: Fit ARIMA model (with or without exogenous variables)
    has_exog = not exog_df.empty and exog_df.shape[1] > 0

    if horizon_settings["mode"] == "D":
        # Short daily horizons: use fixed AR(1) model
        # Ensure seed is set before ARIMA fitting
        np.random.seed(42)
        if not exog_df.empty:
            arima_full = ARIMA(log_returns, order=(1, 0, 0), exog=exog_df).fit()
        else:
            arima_full = ARIMA(log_returns, order=(1, 0, 0)).fit()
        model_type = "arima_fixed"
    else:
        # Longer horizons: auto-select ARIMA order
        # Set random seed before auto_arima for reproducibility
        np.random.seed(42)
        arima_full = auto_arima(
            log_returns,
            exogenous=exog_df if not exog_df.empty else None,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            trace=False,
            random_state=42,  # For reproducibility
            n_jobs=1,  # Single thread for reproducibility
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

    # Step 3: Run backtest to select best model (Random Walk, AR(1), ARIMA, XGBoost)
    best_model_name = "random_walk"
    best_mape = None
    signal_quality = "unknown"
    all_metrics = {}
    
    if len(prices) > forecast_steps * 3 and len(log_returns) > forecast_steps * 3 + 20:
        best_model_name, best_mape, signal_quality, all_metrics = run_backtest(
            log_returns, exog_df, forecast_steps, model_type, prices, use_ml=use_ml
        )

    # Step 4: Generate forecast returns using the selected best model
    if best_model_name == "ar1":
        # Use AR(1) model
        try:
            # Ensure seed is set before AR(1) training
            np.random.seed(42)
            ar1_model = train_ar1(log_returns, exog_df if has_exog else None)
            exog_future = None
            if not exog_df.empty:
                last = exog_df.iloc[-1].values
                exog_future = np.vstack([last] * forecast_steps)
            forecast_returns = forecast_ar1(ar1_model, forecast_steps, exog_future)
        except Exception:
            # Fallback to drift
            forecast_returns = np.full(forecast_steps, drift_full)
    elif best_model_name == "arima":
        exog_future = None
        if not exog_df.empty:
            last = exog_df.iloc[-1].values
            exog_future = np.vstack([last] * forecast_steps)
        forecast_returns = forecast_from_arima(arima_full, model_type, forecast_steps, exog_future)
    elif best_model_name == "random_walk" or best_model_name == "drift":
        # Deterministic expected return forecast (no artificial wiggles).
        # We will simulate day-to-day volatility only for plotting (see plot_forecast).
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
    # ARIMA and AR1 models mean-revert to ~0 which can lead to overly optimistic or flat forecasts at longer horizons.
    # For long horizons, use historical drift (expected value) for the point forecast.
    horizon_days = horizon_settings.get("invested_days", forecast_steps * 30)  # Approximate if not available
    mode = horizon_settings.get("mode", "D")
    is_long_horizon = (
        horizon_days >= 180 or  # 6+ months in any mode (6 months = 126 days, but we want to include it)
        forecast_steps > 20 or  # Long daily/weekly forecasts
        (mode == "M" and forecast_steps >= 3)  # Monthly: 3 months (3 steps) or longer - AR1/ARIMA can be unreliable
    )
    
    # Apply drift-based adjustment for ARIMA and AR1 models on long horizons
    # AR1 models are known to perform poorly on longer horizons (see comment at line 110)
    if best_model_name in ("arima", "ar1") and is_long_horizon:
        adjusted_returns = np.full(forecast_steps, drift_full)
    else:
        adjusted_returns = forecast_returns
    
    # Step 6: Convert log-returns to prices
    cum_returns = np.cumsum(adjusted_returns)
    last_price = float(prices.iloc[-1])
    forecast_prices = last_price * np.exp(cum_returns)
    future_dates = pd.date_range(start=start, periods=forecast_steps, freq=freq)
    forecast_series = pd.Series(forecast_prices, index=future_dates)

    # Step 7: Estimate future volatility using GARCH(1,1)
    #
    # IMPORTANT:
    # `log_returns` may be weekly/monthly (because `prices` is resampled for the selected horizon).
    # Fitting GARCH on resampled returns and then annualizing with sqrt(252) will massively inflate
    # volatility (e.g., >200%) and often produces flat multi-step forecasts.
    #
    # To keep units consistent, we always fit GARCH on DAILY log-returns and then aggregate the
    # predicted daily variance into per-step (daily/weekly/monthly) volatility.
    horizon_days = int(horizon_settings.get("invested_days", forecast_steps))

    # Determine "days per step" for aggregation.
    if horizon_settings["interval"] == "1d":
        days_per_step = 1
    elif horizon_settings["interval"] == "1wk":
        days_per_step = 5
    elif horizon_settings["interval"] == "1mo":
        days_per_step = 21
    else:
        days_per_step = 1

    # Prefer true daily prices if available; otherwise fetch daily series from cache.
    daily_prices: Optional[pd.Series] = None
    if raw_prices is not None and len(raw_prices) > 0:
        try:
            gaps = pd.Series(raw_prices.index).sort_values().diff().dropna()
            median_gap_days = float(gaps.dt.total_seconds().median() / 86400.0) if len(gaps) else 999.0
            if median_gap_days <= 3.5:
                daily_prices = raw_prices.dropna()
        except Exception:
            daily_prices = None

    if daily_prices is None:
        try:
            from .data_loader import fetch_yfinance
            daily_data = fetch_yfinance(ticker, horizon_settings.get("download_period", "5y"), "1d", use_cache=True)
            if not daily_data.empty and "Adj Close" in daily_data.columns:
                daily_prices = daily_data["Adj Close"].dropna()
            elif not daily_data.empty and "Close" in daily_data.columns:
                daily_prices = daily_data["Close"].dropna()
        except Exception:
            daily_prices = None

    if daily_prices is None or daily_prices.empty:
        raise ValueError("Insufficient daily price data to estimate volatility.")

    daily_log_returns = np.log(daily_prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if daily_log_returns.empty:
        raise ValueError("Insufficient daily return data to estimate volatility.")

    returns = 100 * daily_log_returns  # percent daily returns for arch_model stability
    
    try:
        from arch import arch_model
    except Exception as e:
        raise ImportError(
            "The 'arch' package is required to estimate forward volatility but failed to import.\n"
            "To install, use conda:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate stock-forecast\n"
            "Original import error: %s" % e
        ) from e

    # Set random seed before GARCH fitting for reproducibility
    np.random.seed(42)
    garch = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    # Use deterministic optimization settings
    garch_fit = garch.fit(disp="off", options={'maxiter': 1000, 'disp': False})

    # In-sample conditional volatility from the fitted GARCH model (same frequency as training).
    # returns are in percent => conditional_volatility is also in percent (daily).
    # Convert to annualized percent for plotting consistency.
    sigma_fitted: Optional[pd.Series] = None
    try:
        if hasattr(garch_fit, "conditional_volatility"):
            cond_vol = garch_fit.conditional_volatility
            cond_vol_arr = np.asarray(cond_vol, dtype=float)
            sigma_fitted = (
                pd.Series(cond_vol_arr, index=returns.index)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                * np.sqrt(252.0)
            )
    except Exception:
        sigma_fitted = None

    # Forecast DAILY variance for the full horizon in business-day units.
    garch_forecast = garch_fit.forecast(horizon=horizon_days)
    
    # Extract variance forecast correctly
    # The arch library's forecast.variance returns conditional variance forecasts
    # It's typically a DataFrame with shape (1, horizon) where columns are forecast steps
    variance_data = garch_forecast.variance
    
    # Convert to numpy array, handling DataFrame or array input
    if hasattr(variance_data, 'values'):
        var_array = variance_data.values
    elif hasattr(variance_data, 'iloc'):
        var_array = variance_data.iloc[:, :].values
    else:
        var_array = np.asarray(variance_data)
    
    # Flatten to 1D and extract the forecast horizon
    var_flat = var_array.flatten()
    
    # The forecast should have horizon_days values
    # If it's a 2D array (1, horizon_days), take the first row
    # If it's already 1D with horizon_days elements, use it directly
    if var_array.ndim == 2:
        if var_array.shape[0] == 1:
            daily_var_pct2 = var_array[0, :]
        elif var_array.shape[1] == 1:
            daily_var_pct2 = var_array[:, 0]
        else:
            # Take diagonal or first row
            daily_var_pct2 = np.diag(var_array) if var_array.shape[0] == var_array.shape[1] else var_array[0, :]
    else:
        daily_var_pct2 = var_flat
    
    daily_var_pct2 = np.asarray(daily_var_pct2, dtype=float)
    
    # Ensure we have variation in the forecast
    # GARCH(1,1) forecasts should decay from current conditional variance to unconditional variance
    if len(daily_var_pct2) > 1:
        # Check if forecast is too flat (all values nearly identical)
        if np.std(daily_var_pct2) < 1e-6 or np.allclose(daily_var_pct2, daily_var_pct2[0], rtol=1e-5):
            # Reconstruct forecast manually using GARCH dynamics
            # Get current conditional variance
            try:
                if hasattr(garch_fit, 'conditional_volatility'):
                    cond_vol = garch_fit.conditional_volatility
                    if hasattr(cond_vol, 'iloc'):
                        current_cond_var = float(cond_vol.iloc[-1] ** 2)
                    else:
                        current_cond_var = float(cond_vol[-1] ** 2)
                else:
                    # Use recent realized variance
                    current_cond_var = float(returns.iloc[-min(21, len(returns)):].var())
            except Exception:
                current_cond_var = float(daily_var_pct2[0]) if len(daily_var_pct2) > 0 else float(returns.var())
            
            # Get unconditional variance
            try:
                unconditional_var = float(garch_fit.unconditional_variance)
            except Exception:
                # Estimate from long-term average variance
                unconditional_var = float(returns.rolling(window=min(252, len(returns))).var().dropna().mean())
            
            # GARCH(1,1) variance forecast decays exponentially to unconditional variance
            # h_t = omega + alpha * eps^2_{t-1} + beta * h_{t-1}
            # Long-term forecast: h_inf = omega / (1 - alpha - beta)
            # The forecast decays as: h_t = unconditional_var + (current_var - unconditional_var) * (alpha + beta)^t
            try:
                params = garch_fit.params
                # Try different parameter name formats used by arch library
                alpha = None
                beta = None
                if 'alpha[1]' in params.index:
                    alpha = float(params['alpha[1]'])
                elif 'alpha' in params.index:
                    alpha = float(params['alpha'])
                if 'beta[1]' in params.index:
                    beta = float(params['beta[1]'])
                elif 'beta' in params.index:
                    beta = float(params['beta'])
                
                if alpha is not None and beta is not None:
                    persistence = alpha + beta
                else:
                    # Estimate persistence from the forecast itself if available
                    if len(daily_var_pct2) > 1 and daily_var_pct2[0] != daily_var_pct2[-1]:
                        # Estimate decay rate from first to last value
                        ratio = daily_var_pct2[-1] / daily_var_pct2[0] if daily_var_pct2[0] > 0 else 1.0
                        persistence = ratio ** (1.0 / max(1, horizon_days - 1))
                    else:
                        persistence = 0.95  # Default decay rate
            except Exception:
                persistence = 0.95  # Default decay rate
            
            # Create decaying forecast
            t = np.arange(horizon_days)
            decay = persistence ** t
            daily_var_pct2 = unconditional_var + (current_cond_var - unconditional_var) * decay

    # Sanitize variance forecast: replace non-finite values, fill gaps, and clip at 0
    daily_var_pct2 = np.asarray(daily_var_pct2, dtype=float)
    daily_var_pct2[~np.isfinite(daily_var_pct2)] = np.nan
    if daily_var_pct2.size == 0:
        daily_var_pct2 = np.full(horizon_days, np.nan, dtype=float)
    if np.all(np.isnan(daily_var_pct2)):
        # Fallback: realized variance from recent history (units: percent^2)
        fallback_var = float(np.nanvar(returns.iloc[-min(252, len(returns)):]))
        daily_var_pct2 = np.full(horizon_days, fallback_var, dtype=float)
    else:
        daily_var_pct2 = (
            pd.Series(daily_var_pct2)
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
    daily_var_pct2 = np.maximum(daily_var_pct2, 0.0)

    # Create a stochastic daily variance path (percent^2) using GARCH recursion so the
    # forecast looks like a continuation (not a deterministic straight line).
    daily_var_path_pct2: Optional[np.ndarray] = None
    try:
        params = garch_fit.params
        def _param(name: str, default: float) -> float:
            try:
                if hasattr(params, "index") and name in params.index:
                    return float(params[name])
                if isinstance(params, dict) and name in params:
                    return float(params[name])
            except Exception:
                pass
            return float(default)

        omega = _param("omega", 0.0)
        alpha = _param("alpha[1]", _param("alpha", 0.1))
        beta = _param("beta[1]", _param("beta", 0.85))
        persistence = alpha + beta
        if not np.isfinite(persistence) or persistence <= 0:
            alpha, beta = 0.1, 0.85
        if alpha + beta >= 0.999:
            # keep it stable-ish
            beta = min(beta, 0.998)
            alpha = min(alpha, 0.998 - beta)

        # Start from last conditional variance (percent^2)
        try:
            if hasattr(garch_fit, "conditional_volatility"):
                cv = garch_fit.conditional_volatility
                last_cv = float(cv.iloc[-1] if hasattr(cv, "iloc") else cv[-1])
                h = max(last_cv * last_cv, 0.0)
            else:
                h = float(np.nanvar(returns.iloc[-min(252, len(returns)):]))
        except Exception:
            h = float(np.nanvar(returns))

        rng = np.random.default_rng(42)
        path = np.empty(horizon_days, dtype=float)
        for t in range(horizon_days):
            z = float(rng.standard_normal())
            eps = np.sqrt(max(h, 0.0)) * z
            h = omega + alpha * (eps ** 2) + beta * h
            if not np.isfinite(h) or h < 0:
                h = float(np.nanmean(daily_var_pct2)) if np.isfinite(np.nanmean(daily_var_pct2)) else 0.0
            path[t] = h
        daily_var_path_pct2 = path
    except Exception:
        daily_var_path_pct2 = None
    
    # Ensure we have the right length
    if daily_var_pct2.size < horizon_days:
        # Pad with last value (GARCH converges to unconditional variance)
        pad = np.full(horizon_days - daily_var_pct2.size, float(daily_var_pct2[-1]) if daily_var_pct2.size else 0.0)
        daily_var_pct2 = np.concatenate([daily_var_pct2, pad])
    elif daily_var_pct2.size > horizon_days:
        # Trim to horizon_days
        daily_var_pct2 = daily_var_pct2[:horizon_days]

    # Aggregate daily variance into each forecast "step" (day/week/month).
    # - period_std_decimal is used for Monte Carlo shocks (consistent with step frequency)
    # - sigma_forecast is for plotting/reporting (annualized %)
    period_std_decimal = []
    sigma_ann_pct = []
    for i in range(forecast_steps):
        a = i * days_per_step
        b = min((i + 1) * days_per_step, horizon_days)
        var_for_points = daily_var_path_pct2 if daily_var_path_pct2 is not None else daily_var_pct2
        chunk = var_for_points[a:b]
        if chunk.size == 0:
            chunk = var_for_points[-1:]
        # Convert daily %^2 -> daily decimal variance
        daily_var_dec = (chunk / (100.0 ** 2))
        period_var_dec = float(np.nansum(daily_var_dec))  # variance adds over days
        period_std_decimal.append(float(np.sqrt(max(period_var_dec, 0.0))))

        mean_daily_var_dec = float(np.nanmean(daily_var_dec))
        if not np.isfinite(mean_daily_var_dec):
            mean_daily_var_dec = 0.0
        sigma_ann_pct.append(float(np.sqrt(max(mean_daily_var_dec, 0.0) * 252.0) * 100.0))

    period_std_decimal = np.asarray(period_std_decimal, dtype=float)
    sigma_forecast = pd.Series(sigma_ann_pct, index=future_dates)

    # Also keep a daily-resolution volatility forecast for plotting (annualized %).
    # Use the stochastic path when available to avoid a deterministic straight line.
    try:
        start_daily = pd.Timestamp(daily_prices.index[-1]) + pd.offsets.BDay()
        daily_future_dates = pd.bdate_range(start=start_daily, periods=horizon_days)
        var_for_daily = daily_var_path_pct2 if daily_var_path_pct2 is not None else daily_var_pct2
        sigma_daily_forecast = pd.Series(
            np.sqrt(np.maximum(np.asarray(var_for_daily[:horizon_days], dtype=float), 0.0)) * np.sqrt(252.0),
            index=daily_future_dates,
        ).replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        sigma_daily_forecast = None

    # Soft adjustment used for uncertainty bands / MC (keep sigma_forecast as the raw model output).
    vol_scale = 1.25 if forecast_steps > 20 else 1.0
    period_std_soft_decimal = period_std_decimal * vol_scale

    # Step 8: Monte Carlo simulation for probabilistic forecasts
    num_sims = int(num_sims)
    np.random.seed(42)  # For reproducibility
    
    # Volatility for MC must match the step frequency of `adjusted_returns` (daily/weekly/monthly).
    sigma_for_mc = period_std_soft_decimal
    
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
        # Confidence bounds consistent with step frequency:
        # impact is proportional to period std of returns (decimal).
        impact_vol = last_price * pd.Series(period_std_soft_decimal, index=future_dates)
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
        "sigma_daily_forecast": sigma_daily_forecast,
        "sigma_fitted": sigma_fitted,
        "last_price": last_price,
        "forecast_returns": forecast_returns,
        "all_metrics": all_metrics,  # Include all metrics for CSV export
    }


def plot_forecast(ticker: str, prices: pd.Series, raw_prices: pd.Series, forecast_artifacts: Dict[str, object], save: bool = False, horizon_suffix: str = "") -> Optional[str]:
    """Plot the forecast and optionally save to `results/`.

    Args:
        ticker: Stock symbol
        prices: Historical price series (may be resampled)
        raw_prices: Raw price series (may also be resampled for monthly/weekly forecasts)
        forecast_artifacts: Dictionary with forecast data
        save: Whether to save the plot
        horizon_suffix: Optional suffix to add to filename (e.g., "10_days", "3_months")
    
    Returns the path to the saved image if `save=True`, otherwise None.
    """
    forecast_series = forecast_artifacts["forecast_series"]
    hybrid_upper = forecast_artifacts["hybrid_upper"]
    hybrid_lower = forecast_artifacts["hybrid_lower"]
    mc_p10 = forecast_artifacts["mc_p10"]
    mc_p50 = forecast_artifacts["mc_p50"]
    mc_p90 = forecast_artifacts["mc_p90"]
    sigma_forecast = forecast_artifacts.get("sigma_forecast")
    forecast_returns = forecast_artifacts.get("forecast_returns")

    plt.figure(figsize=(10, 5))
    
    # Load original daily data for plotting (not resampled)
    # raw_prices may be resampled for monthly/weekly forecasts, so fetch daily data
    try:
        from .data_loader import fetch_yfinance
        # Fetch daily data from cache (should be available)
        daily_data = fetch_yfinance(ticker, "5y", "1d", use_cache=True)
        if not daily_data.empty and "Adj Close" in daily_data.columns:
            daily_prices = daily_data["Adj Close"].dropna()
        else:
            daily_prices = None
    except Exception:
        daily_prices = None
    
    # Use daily_prices if available, otherwise fall back to raw_prices or prices
    if daily_prices is not None and len(daily_prices) > 0:
        plot_prices = daily_prices
    elif raw_prices is not None and len(raw_prices) > 0:
        plot_prices = raw_prices
    else:
        plot_prices = prices
    
    forecast_length = len(forecast_series)
    
    # Get the last date from plot_prices (original daily data)
    if plot_prices is not None and len(plot_prices) > 0:
        last_date = plot_prices.index[-1]
        forecast_end = forecast_series.index[-1] if len(forecast_series) > 0 else last_date
        
        # Determine lookback period based on forecast horizon
        # Show enough history to see context, but not so much that forecast is invisible
        if forecast_length <= 10:
            # 10 days: show last 3 months
            lookback_days = 90
        elif forecast_length <= 3:
            # 1-3 months: show last 12 months
            lookback_days = 365
        elif forecast_length <= 6:
            # 3-6 months: show last 18 months
            lookback_days = 540
        else:
            # 1 year: show last 2 years
            lookback_days = 730
        
        start_date = last_date - pd.Timedelta(days=lookback_days)
        
        # Filter plot_prices (original daily data) by start date
        # Keep original frequency - no resampling
        historical_prices = plot_prices[plot_prices.index >= start_date]
        
        # Ensure we have at least a few data points
        if len(historical_prices) < 3:
            historical_prices = plot_prices
    else:
        # Fallback to resampled prices if raw_prices not available
        historical_prices = prices
        last_date = prices.index[-1] if len(prices) > 0 else None
        forecast_end = forecast_series.index[-1] if len(forecast_series) > 0 else None
        start_date = prices.index[0] if len(prices) > 0 else None
    
    # Plot historical prices using original daily frequency
    plt.plot(historical_prices.index, historical_prices, label="Historical Prices", color="tab:blue", linewidth=1.5)
    
    # Connect last historical point to forecast for better visualization
    # Use the last price from plot_prices (daily data) or fallback
    if plot_prices is not None and len(plot_prices) > 0:
        last_price = float(plot_prices.iloc[-1])
        last_date = plot_prices.index[-1]
    elif raw_prices is not None and len(raw_prices) > 0:
        last_price = float(raw_prices.iloc[-1])
        last_date = raw_prices.index[-1]
    else:
        last_price = float(prices.iloc[-1])
        last_date = prices.index[-1]
    # --- Daily volatility simulation for plotting (to avoid linear look on monthly/weekly horizons) ---
    def _stable_seed_from_text(text: str) -> int:
        import hashlib
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

    def _simulate_daily_path_for_daily_forecast(
        start_price: float,
        start_dt: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        forecast_prices: pd.Series,
        period_sigma_annualized_pct: Optional[pd.Series],
        seed_text: str,
    ) -> Optional[pd.Series]:
        """Simulate a daily path for daily forecasts by adding volatility to each day.
        
        This is simpler than the period endpoint matching function and works better
        for daily forecasts where each period is just one day.
        """
        if forecast_dates is None or len(forecast_dates) == 0:
            return None
        
        np.random.seed(_stable_seed_from_text(seed_text))
        
        # Generate all business days from start to end of forecast
        all_days = pd.bdate_range(start=start_dt + pd.offsets.BDay(), end=forecast_dates[-1])
        if len(all_days) == 0:
            return None
        
        current_price = float(start_price)
        out_dates = []
        out_prices = []
        
        # Get daily volatility (use first value if available, or average)
        if period_sigma_annualized_pct is not None and len(period_sigma_annualized_pct) > 0:
            avg_vol_ann = float(period_sigma_annualized_pct.mean())
        else:
            avg_vol_ann = 20.0  # Default 20% annualized volatility
        
        sigma_daily = (avg_vol_ann / 100.0) / np.sqrt(252)
        
        # Calculate target log return for the full period
        target_price = float(forecast_prices.iloc[-1])
        target_lr_total = float(np.log(target_price / start_price))
        
        # Distribute the total return across all days with daily volatility
        n_days = len(all_days)
        drift_daily = target_lr_total / n_days if n_days > 0 else 0.0
        
        # Generate daily returns with volatility
        daily_shocks = np.random.normal(loc=0.0, scale=sigma_daily, size=n_days)
        daily_returns = drift_daily + daily_shocks
        
        # Brownian-bridge correction to ensure we hit the target price
        actual_sum = float(daily_returns.sum())
        if abs(actual_sum - target_lr_total) > 1e-10:
            correction = (target_lr_total - actual_sum) / n_days
            daily_returns = daily_returns + correction
        
        # Build price path
        for d, r in zip(all_days, daily_returns):
            current_price = current_price * float(np.exp(r))
            out_dates.append(d)
            out_prices.append(current_price)
        
        if not out_dates:
            return None
        return pd.Series(out_prices, index=pd.DatetimeIndex(out_dates), name="simulated_daily_path")

    def _simulate_daily_path_matching_period_endpoints(
        start_price: float,
        start_dt: pd.Timestamp,
        period_end_dates: pd.DatetimeIndex,
        period_end_prices: pd.Series,
        period_log_returns: Optional[np.ndarray],
        period_sigma_annualized_pct: Optional[pd.Series],
        seed_text: str,
    ) -> Optional[pd.Series]:
        """Simulate a daily path that *hits* each period endpoint price while exhibiting daily volatility.

        Uses a Brownian-bridge-style correction per period so the simulated daily returns sum exactly to the
        target period log return.
        """
        if period_end_dates is None or len(period_end_dates) == 0:
            return None

        np.random.seed(_stable_seed_from_text(seed_text))

        boundaries = [pd.Timestamp(start_dt)] + [pd.Timestamp(d) for d in period_end_dates]
        current_price = float(start_price)
        out_dates = []
        out_prices = []

        for i in range(len(period_end_dates)):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]

            # Business days from next business day after seg_start up to seg_end inclusive
            seg_days = pd.bdate_range(seg_start + pd.offsets.BDay(), seg_end)
            n = len(seg_days)
            if n <= 0:
                continue

            # Target period log return
            if period_log_returns is not None and len(period_log_returns) > i:
                target_lr = float(period_log_returns[i])
            else:
                # Derive from endpoint prices if returns not available
                prev_endpoint = float(start_price) if i == 0 else float(period_end_prices.iloc[i - 1])
                curr_endpoint = float(period_end_prices.iloc[i])
                if prev_endpoint <= 0 or curr_endpoint <= 0:
                    target_lr = 0.0
                else:
                    target_lr = float(np.log(curr_endpoint / prev_endpoint))

            # Daily volatility in log-return terms (annualized % -> daily sigma)
            if period_sigma_annualized_pct is not None and len(period_sigma_annualized_pct) > i:
                vol_ann = float(period_sigma_annualized_pct.iloc[i])
                sigma_daily = (vol_ann / 100.0) / np.sqrt(252)
            else:
                sigma_daily = 0.0

            shocks = np.random.normal(loc=0.0, scale=sigma_daily, size=n)
            drift_daily = target_lr / n
            # Brownian-bridge correction so sum(daily_returns) == target_lr exactly
            correction = (target_lr - (drift_daily * n + float(shocks.sum()))) / n
            daily_logr = drift_daily + shocks + correction

            # Build prices
            for d, r in zip(seg_days, daily_logr):
                current_price = current_price * float(np.exp(r))
                out_dates.append(d)
                out_prices.append(current_price)

        if not out_dates:
            return None
        return pd.Series(out_prices, index=pd.DatetimeIndex(out_dates), name="simulated_daily_path")

    # Decide if we should draw a daily-volatility path (weekly/monthly horizons look linear otherwise)
    # Also simulate for daily forecasts when returns are constant (e.g., random_walk model)
    # For short daily forecasts (≤15 days), always simulate to show realistic day-to-day fluctuations
    infer = None
    try:
        infer = pd.infer_freq(forecast_series.index)
    except Exception:
        infer = None
    is_period_horizon = infer in ("MS", "W-FRI") or ("month" in horizon_suffix) or ("year" in horizon_suffix) or ("week" in horizon_suffix)
    
    # Check if forecast returns are constant (which would make the plot linear)
    returns_are_constant = False
    if forecast_returns is not None:
        forecast_returns_array = forecast_returns if isinstance(forecast_returns, np.ndarray) else np.array(forecast_returns)
        returns_are_constant = len(forecast_returns_array) > 0 and np.allclose(forecast_returns_array, forecast_returns_array[0], rtol=1e-10)
    
    # Simulate daily volatility for:
    # 1. Period horizons (weekly/monthly) - always simulate
    # 2. Daily forecasts with constant returns - simulate to avoid linear appearance
    # 3. Short daily forecasts (≤15 days) - always simulate to show realistic fluctuations
    is_daily_forecast = (infer == "B" or infer is None) and not is_period_horizon
    is_short_daily = is_daily_forecast and forecast_length <= 15
    should_simulate_daily = is_period_horizon or (is_daily_forecast and (returns_are_constant or is_short_daily))

    simulated_daily = None
    if should_simulate_daily:
        if is_daily_forecast and (returns_are_constant or is_short_daily):
            # Use simpler daily simulation for daily forecasts (especially when returns are constant or forecast is short)
            simulated_daily = _simulate_daily_path_for_daily_forecast(
                start_price=last_price,
                start_dt=last_date,
                forecast_dates=forecast_series.index,
                forecast_prices=forecast_series,
                period_sigma_annualized_pct=sigma_forecast,
                seed_text=f"{ticker}|{horizon_suffix}|plot_daily_sim",
            )
        else:
            # Use period endpoint matching for weekly/monthly forecasts
            simulated_daily = _simulate_daily_path_matching_period_endpoints(
                start_price=last_price,
                start_dt=last_date,
                period_end_dates=forecast_series.index,
                period_end_prices=forecast_series,
                period_log_returns=forecast_returns,
                period_sigma_annualized_pct=sigma_forecast,
                seed_text=f"{ticker}|{horizon_suffix}|plot_daily_sim",
            )

    # We still need these for confidence bands (defined regardless of plotting branch)
    extended_dates = [last_date] + list(forecast_series.index)
    extended_forecast = [last_price] + list(forecast_series.values)

    if simulated_daily is not None and len(simulated_daily) > 0:
        plt.plot(simulated_daily.index, simulated_daily.values, color="red", linewidth=1.6, label="Simulated Daily Path (volatility)")
        # Plot point forecasts as markers at period endpoints (no straight connecting line)
        plt.plot(forecast_series.index, forecast_series.values, color="red", marker="o", linestyle="None", markersize=5, label="Point Forecast (period)")
        # Connect last historical point to first forecast point subtly
        plt.plot([last_date, forecast_series.index[0]], [last_price, float(forecast_series.iloc[0])], color="red", linestyle=":", linewidth=1.0)
    else:
        # Default behavior: connect last historical point to forecast
        plt.plot(extended_dates, extended_forecast, label="Forecasted Prices", color="red", marker="o", markersize=5)
    
    # Set x-axis limits to zoom in on relevant period (all forecasts)
    if len(historical_prices) > 0:
        plot_start = historical_prices.index[0]
        plot_end = forecast_end if forecast_end is not None else historical_prices.index[-1]
        # Add some padding
        padding_days = max(7, forecast_length * 2)
        xlim_start = plot_start - pd.Timedelta(days=padding_days)
        xlim_end = plot_end + pd.Timedelta(days=padding_days)
        plt.xlim(xlim_start, xlim_end)
    
    plt.title(f"{ticker} Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    # Extend confidence bounds to connect with last price
    extended_upper = [last_price] + list(hybrid_upper.values)
    extended_lower = [last_price] + list(hybrid_lower.values)
    
    plt.fill_between(
        extended_dates,
        extended_lower,
        extended_upper,
        color="gray",
        alpha=0.3,
        label="95% Confidence Interval",
    )
    plt.plot(extended_dates, extended_upper, linestyle="--", color="orange", label="Upper Confidence Bound")
    plt.plot(extended_dates, extended_lower, linestyle="--", color="green", label="Lower Confidence Bound")
    
    # Extend MC bands
    extended_mc_p10 = [last_price] + list(mc_p10.values)
    extended_mc_p90 = [last_price] + list(mc_p90.values)
    extended_mc_p50 = [last_price] + list(mc_p50.values)
    
    plt.fill_between(extended_dates, extended_mc_p10, extended_mc_p90, color="orange", alpha=0.2, label="MC 10-90%")
    plt.plot(extended_dates, extended_mc_p50, color="orange", linestyle=":", label="MC Median", marker='x')
    plt.legend()
    plt.grid()
    # Add a horizon uncertainty warning for long horizons
    try:
        if len(forecast_series) > 30:
            plt.gcf().text(0.02, 0.95, "Warning: longer horizons have larger uncertainty; interpret probabilistically.", fontsize=9, color="red")
    except Exception:
        pass

    saved_path = None
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if horizon_suffix:
            filename = f"forecast_{ticker}_{horizon_suffix}_{timestamp}.png"
        else:
            filename = f"forecast_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight")
    plt.close()  # Close figure instead of showing (non-blocking)
    return saved_path


def plot_volatility_forecast(
    ticker: str,
    log_returns: pd.Series,
    sigma_forecast: pd.Series,
    save: bool = False,
    horizon_suffix: str = "",
    raw_prices: Optional[pd.Series] = None,
    sigma_daily_forecast: Optional[pd.Series] = None,
    sigma_fitted: Optional[pd.Series] = None
) -> Optional[str]:
    """Plot historical volatility vs GARCH forecast.
    
    Args:
        ticker: Stock symbol
        log_returns: Historical log returns series (may be resampled for weekly/monthly horizons)
        sigma_forecast: GARCH forecasted volatility (annualized percentage)
        save: Whether to save the plot
        horizon_suffix: Optional suffix to add to filename (e.g., "10_days", "3_months")
        raw_prices: Optional raw daily prices (used to calculate accurate daily historical volatility)
        
    Returns:
        Path to saved file if save=True, otherwise None
    """
    # Calculate historical rolling volatility from *daily* data.
    # Important: for monthly/weekly horizons, yfinance may return monthly/weekly prices,
    # so `raw_prices` may NOT be daily. In that case, pull daily prices for this plot.
    daily_prices: Optional[pd.Series] = None
    if raw_prices is not None and len(raw_prices) > 0:
        try:
            # Use median day gap as a robust "daily-ish" detector (infer_freq often returns None).
            gaps = pd.Series(raw_prices.index).sort_values().diff().dropna()
            median_gap_days = float(gaps.dt.total_seconds().median() / 86400.0) if len(gaps) else 999.0
            if median_gap_days <= 3.5:  # business-day-ish
                daily_prices = raw_prices
        except Exception:
            daily_prices = None

    if daily_prices is None:
        try:
            from .data_loader import fetch_yfinance
            daily_data = fetch_yfinance(ticker, "5y", "1d", use_cache=True)
            if not daily_data.empty and "Adj Close" in daily_data.columns:
                daily_prices = daily_data["Adj Close"].dropna()
            elif not daily_data.empty and "Close" in daily_data.columns:
                daily_prices = daily_data["Close"].dropna()
        except Exception:
            daily_prices = None

    if daily_prices is not None and len(daily_prices) > 21:
        daily_log_returns = np.log(daily_prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily_log_returns) > 21:
            rolling_vol = daily_log_returns.rolling(window=21).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
        else:
            rolling_vol = pd.Series(dtype=float)
    elif len(log_returns) > 21:
        # Fallback: use provided log_returns, but need to determine frequency
        # Infer frequency from the data
        try:
            freq = pd.infer_freq(log_returns.index)
            if freq and 'W' in freq:
                # Weekly data: annualize with sqrt(52)
                annualization_factor = np.sqrt(52)
            elif freq and 'M' in freq or freq == 'MS':
                # Monthly data: annualize with sqrt(12)
                annualization_factor = np.sqrt(12)
            else:
                # Assume daily: annualize with sqrt(252)
                annualization_factor = np.sqrt(252)
        except Exception:
            # Default to daily if can't infer
            annualization_factor = np.sqrt(252)
        
        rolling_vol = log_returns.rolling(window=21).std() * annualization_factor * 100
        rolling_vol = rolling_vol.dropna()
    else:
        rolling_vol = pd.Series(dtype=float)
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical volatility:
    # Prefer the in-sample conditional volatility from the fitted GARCH model (same frequency as training)
    # when provided; otherwise fall back to rolling realized volatility.
    hist_label = "Historical Volatility (21-day rolling)"
    hist_series = rolling_vol
    if sigma_fitted is not None:
        try:
            sigma_fitted_clean = sigma_fitted.replace([np.inf, -np.inf], np.nan).dropna()
            if len(sigma_fitted_clean) > 10:
                hist_series = sigma_fitted_clean
                hist_label = "GARCH Fitted Volatility (in-sample, annualized)"
        except Exception:
            pass

    if hist_series is not None and not hist_series.empty:
        plt.plot(
            hist_series.index,
            hist_series.values,
            label=hist_label,
            color="blue",
            alpha=0.75,
            linewidth=1.6,
            zorder=2,
        )
    
    # Plot GARCH forecast:
    # - daily curve (if available) to show shape
    # - period-end markers from sigma_forecast for the specific horizon
    if sigma_daily_forecast is not None and len(sigma_daily_forecast) > 0:
        sigma_daily_clean = sigma_daily_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        sigma_points = sigma_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        if (sigma_points is None or sigma_points.empty) and sigma_daily_clean is not None and not sigma_daily_clean.empty:
            # Derive horizon points from daily curve if the period series is missing/NaN
            try:
                sigma_points = sigma_daily_clean.reindex(sigma_forecast.index, method="ffill").dropna()
            except Exception:
                sigma_points = sigma_daily_clean.iloc[:: max(1, int(len(sigma_daily_clean) / max(1, len(sigma_forecast))))].copy()

        plt.plot(
            sigma_daily_clean.index,
            sigma_daily_clean.values,
            label="GARCH(1,1) Forecast (daily, annualized)",
            color="red",
            linewidth=2,
            alpha=0.85,
            zorder=3,
        )
        plt.plot(
            sigma_points.index,
            sigma_points.values,
            label="GARCH Forecast (horizon points)",
            color="darkred",
            linewidth=0,
            marker="o",
            markersize=7,
            zorder=5,
        )
    else:
        sigma_points = sigma_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        plt.plot(
            sigma_points.index,
            sigma_points.values,
            label="GARCH(1,1) Forecast (annualized)",
            color="red",
            linewidth=2,
            marker="o",
            markersize=6,
            zorder=4,
        )
    
    # Add vertical line separating historical from forecast
    if not rolling_vol.empty:
        last_hist_date = rolling_vol.index[-1]
        plt.axvline(x=last_hist_date, color="gray", linestyle="--", alpha=0.5, label="Forecast Start")
    
    # Set x-axis limits adapted to the forecast horizon
    # For short horizons (10 days, 1 month), show less history to make forecast visible
    # For longer horizons, show more history proportionally
    if not sigma_forecast.empty:
        forecast_start = sigma_forecast.index[0]
        forecast_end = sigma_forecast.index[-1]
        forecast_duration = (forecast_end - forecast_start).days
        
        # Handle edge case: if forecast has only one point or duration is 0,
        # estimate duration from number of steps
        if forecast_duration <= 0 and len(sigma_forecast) > 1:
            # Estimate from the frequency of the index
            try:
                freq = pd.infer_freq(sigma_forecast.index)
                if freq and 'D' in freq:
                    forecast_duration = len(sigma_forecast)  # daily
                elif freq and 'W' in freq:
                    forecast_duration = len(sigma_forecast) * 7  # weekly
                elif freq and 'M' in freq:
                    forecast_duration = len(sigma_forecast) * 30  # monthly (approx)
                else:
                    forecast_duration = len(sigma_forecast) * 30  # default to monthly
            except Exception:
                forecast_duration = len(sigma_forecast) * 30  # fallback
        elif forecast_duration <= 0:
            forecast_duration = 10  # default for single point forecasts
        
        # Determine how much history to show based on forecast horizon
        # Use a multiplier approach: show N times the forecast duration as history
        if forecast_duration <= 10:
            # 10 days: show ~3 months of history (90 days) for better context
            history_days = 90
        elif forecast_duration <= 30:
            # 1 month: show ~4 months of history (120 days) for better context
            history_days = 120
        elif forecast_duration <= 90:
            # 3 months: show ~6 months of history
            history_days = 180
        elif forecast_duration <= 180:
            # 6 months: show ~1 year of history
            history_days = 365
        else:
            # 1 year or more: show ~2 years of history
            history_days = 730
        
        # Calculate x-axis limits
        xlim_start = forecast_start - pd.Timedelta(days=history_days)
        xlim_end = forecast_end + pd.Timedelta(days=max(7, forecast_duration // 4))
        
        # Ensure we don't go before available historical data
        # Check both rolling_vol and hist_series (which may be sigma_fitted)
        earliest_date = None
        if not rolling_vol.empty:
            earliest_date = rolling_vol.index[0]
        if hist_series is not None and not hist_series.empty:
            hist_start = hist_series.index[0]
            if earliest_date is None or hist_start < earliest_date:
                earliest_date = hist_start
        
        if earliest_date is not None and xlim_start < earliest_date:
            xlim_start = earliest_date
        
        plt.xlim(xlim_start, xlim_end)
    
    plt.title(f"{ticker} Volatility Forecast: Historical vs GARCH(1,1)")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Annualized %)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    saved_path = None
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if horizon_suffix:
            filename = f"volatility_forecast_{ticker}_{horizon_suffix}_{timestamp}.png"
        else:
            filename = f"volatility_forecast_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
    plt.close()  # Close figure instead of showing (non-blocking)
    return saved_path


def save_model_comparison_csv(
    all_metrics: Dict[str, Dict[str, float]],
    ticker: str,
    horizon_label: str,
    horizon_suffix: str = ""
) -> str:
    """Save model comparison metrics to CSV file.
    
    Args:
        all_metrics: Dictionary with model names as keys and metrics dicts as values
        ticker: Stock symbol
        horizon_label: Horizon label (e.g., "10 days")
        
    Returns:
        Path to saved CSV file
    """
    if not all_metrics:
        return ""
    
    # Create DataFrame
    rows = []
    baseline_mape = None
    if "random_walk" in all_metrics:
        baseline_mape = all_metrics["random_walk"]["MAPE"]
    
    for model_name, metrics in all_metrics.items():
        if model_name == "random_walk":
            beats_baseline = "Baseline"
        elif baseline_mape and metrics["MAPE"] < baseline_mape:
            beats_baseline = "Yes"
        else:
            beats_baseline = "No"
        rows.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "Beats_Baseline": beats_baseline,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("MAPE")  # Sort by MAPE (best first)
    
    # Save to CSV
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if horizon_suffix:
        filename = f"model_comparison_{ticker}_{horizon_suffix}_{timestamp}.csv"
    else:
        filename = f"model_comparison_{ticker}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)
    
    return filepath


def clean_old_results(ticker: str):
    """Delete all old result files for a given ticker before generating new ones.
    
    This function removes all existing forecast plots, volatility plots, and CSV files
    for the specified ticker, ensuring only the current run's results are kept.
    
    Args:
        ticker: Stock symbol to clean results for
    """
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        return
    
    # Patterns for files to clean
    patterns = [
        f"forecast_{ticker}_*.png",
        f"volatility_forecast_{ticker}_*.png",
        f"model_comparison_{ticker}_*.csv",
        f"eda_*_{ticker}_*.png",
        f"eda_statistics_{ticker}_*.txt",
    ]
    
    for pattern in patterns:
        filepath_pattern = os.path.join(results_dir, pattern)
        old_files = glob.glob(filepath_pattern)
        # Delete all old files for this ticker
        for old_file in old_files:
            try:
                os.remove(old_file)
            except Exception:
                pass  # Ignore errors when deleting old files
