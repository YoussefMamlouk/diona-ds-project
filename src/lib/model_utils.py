"""
Model utilities for backtesting and evaluation.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error


def run_backtest(
    log_returns: pd.Series,
    exog_df: pd.DataFrame,
    forecast_periods: int,
    model_type: str,
    prices: pd.Series,
) -> Tuple[str, Optional[float], str]:
    """Run backtests on multiple models and return best performer.
    
    Args:
        log_returns: Log returns series
        exog_df: Exogenous features DataFrame
        forecast_periods: Number of periods to forecast
        model_type: Type of model ('arima_fixed' or 'auto_arima')
        prices: Price series for computing actual values
    
    Returns:
        Tuple of (best_model_name, best_mape, signal_quality)
    """
    if len(log_returns) < forecast_periods * 2 + 10:
        return "arima", None, "unknown"
    
    # Split into train/test
    train_size = len(log_returns) - forecast_periods
    train_returns = log_returns.iloc[:train_size]
    test_returns = log_returns.iloc[train_size:]
    
    has_exog = not exog_df.empty and exog_df.shape[1] > 0
    train_exog = exog_df.iloc[:train_size] if has_exog else None
    test_exog = exog_df.iloc[train_size:] if has_exog else None
    
    results = {}
    
    # Test ARIMA - convert to price level for MAPE calculation
    try:
        if has_exog and train_exog is not None:
            arima_model = ARIMA(train_returns, order=(1, 0, 0), exog=train_exog).fit()
            forecast_returns = arima_model.forecast(steps=forecast_periods, exog=test_exog).values
        else:
            arima_model = ARIMA(train_returns, order=(1, 0, 0)).fit()
            forecast_returns = arima_model.forecast(steps=forecast_periods).values
        
        # Convert log returns to prices for MAPE calculation
        last_price = prices.iloc[train_size - 1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = prices.iloc[train_size:train_size + forecast_periods].values
        
        mape = float(mean_absolute_percentage_error(actual_prices, forecast_prices) * 100)
        results["arima"] = mape
    except Exception:
        pass
    
    # Test simple drift model (mean return) - convert to price level
    try:
        drift = float(train_returns.mean())
        forecast_returns = np.full(forecast_periods, drift)
        
        # Convert to prices
        last_price = prices.iloc[train_size - 1]
        forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = prices.iloc[train_size:train_size + forecast_periods].values
        
        mape = float(mean_absolute_percentage_error(actual_prices, forecast_prices) * 100)
        results["drift"] = mape
    except Exception:
        pass
    
    # Test XGBoost if available - convert to price level
    try:
        from ..models import forecast_with_xgb, train_xgb_cv
        
        xgb_model = train_xgb_cv(train_returns, train_exog)
        if xgb_model is not None:
            forecast_returns = forecast_with_xgb(train_returns, train_exog, forecast_periods, model=xgb_model)
            if forecast_returns is not None:
                # Convert to prices
                last_price = prices.iloc[train_size - 1]
                forecast_prices = last_price * np.exp(np.cumsum(forecast_returns))
                actual_prices = prices.iloc[train_size:train_size + forecast_periods].values
                
                mape = float(mean_absolute_percentage_error(actual_prices, forecast_prices) * 100)
                results["xgb"] = mape
    except Exception:
        pass
    
    if not results:
        return "arima", None, "unknown"
    
    # Find best model
    best_model = min(results, key=results.get)
    best_mape = results[best_model]
    
    # Determine signal quality
    if best_mape < 5:
        signal_quality = "high"
    elif best_mape < 15:
        signal_quality = "medium"
    else:
        signal_quality = "low"
    
    return best_model, best_mape, signal_quality
