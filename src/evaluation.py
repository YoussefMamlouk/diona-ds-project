"""
Evaluation module: model training, backtesting, forecasting, and Monte Carlo simulation.

This module coordinates the forecasting pipeline by:
- Training ARIMA/XGBoost models on historical data
- Running backtests to select the best performer
- Generating forecasts with uncertainty quantification via Monte Carlo
- Adapting to long vs short horizon characteristics
"""
from .models import (
    forecast_from_arima,
    forecast_with_xgb,
    train_ar1,
    forecast_ar1,
)
from .backtests import run_backtest
from .volatility import backtest_garch_volatility
from .plots import plot_forecast, plot_volatility_forecast, plot_volatility_backtest
from .results import save_model_comparison_csv, clean_old_results
import numpy as np
import pandas as pd
from typing import Dict, Optional
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


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

    # Step 3: Run backtest to select best model on validation, then evaluate on test
    best_model_name = "random_walk"
    best_mape = None
    signal_quality = "unknown"
    all_metrics = {}          # test metrics (final, after refit train+val)
    validation_metrics = {}   # selection metrics (validation)
    
    if len(prices) > forecast_steps * 3 and len(log_returns) > forecast_steps * 3 + 20:
        best_model_name, best_mape, signal_quality, all_metrics, validation_metrics = run_backtest(
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
            # Set seed before XGBoost training for reproducibility
            np.random.seed(42)
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

    vol_backtest = backtest_garch_volatility(daily_log_returns, horizon_days)
    
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
    # Prefer asymmetric GARCH with heavy tails; fall back to standard GARCH if needed.
    try:
        garch = arch_model(returns, vol="GARCH", p=1, o=1, q=1, dist="t")
        garch_fit = garch.fit(disp="off", options={'maxiter': 1000, 'disp': False})
    except Exception:
        garch = arch_model(returns, vol="GARCH", p=1, o=0, q=1, dist="normal")
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
                gamma = None
                if 'alpha[1]' in params.index:
                    alpha = float(params['alpha[1]'])
                elif 'alpha' in params.index:
                    alpha = float(params['alpha'])
                if 'beta[1]' in params.index:
                    beta = float(params['beta[1]'])
                elif 'beta' in params.index:
                    beta = float(params['beta'])
                if 'gamma[1]' in params.index:
                    gamma = float(params['gamma[1]'])
                elif 'gamma' in params.index:
                    gamma = float(params['gamma'])
                
                if alpha is not None and beta is not None:
                    gamma_adj = 0.5 * gamma if gamma is not None else 0.0
                    persistence = alpha + beta + gamma_adj
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
        gamma = _param("gamma[1]", _param("gamma", 0.0))
        persistence = alpha + beta + 0.5 * gamma
        if not np.isfinite(persistence) or persistence <= 0:
            alpha, beta, gamma = 0.1, 0.85, 0.0
        if alpha + beta + 0.5 * gamma >= 0.999:
            # keep it stable-ish
            beta = min(beta, 0.998)
            alpha = min(alpha, 0.998 - beta)
            gamma = min(gamma, max(0.0, 1.0 - alpha - beta) * 2.0)

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
            neg = 1.0 if eps < 0 else 0.0
            h = omega + alpha * (eps ** 2) + gamma * neg * (eps ** 2) + beta * h
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
        "vol_backtest": vol_backtest,
        "last_price": last_price,
        "forecast_returns": forecast_returns,
        "all_metrics": all_metrics,  # test metrics (final)
        "validation_metrics": validation_metrics,  # selection metrics
    }
