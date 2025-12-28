"""
Backtesting utilities for return forecasting models.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

from .models import (
    forecast_from_arima,
    forecast_with_xgb,
    train_ar1,
    forecast_ar1,
    train_linear_cv,
    forecast_with_linear,
)


def run_backtest(
    log_returns: pd.Series,
    exog_df: pd.DataFrame,
    forecast_periods: int,
    model_type: str,
    prices: pd.Series,
    use_ml: bool = True,
) -> Tuple[
    str,
    Optional[float],
    str,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
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
        Tuple of:
        - best_model_name
        - best_test_mape (float)
        - signal_quality (based on test MAPE)
        - test_metrics_by_model (RMSE/MAE/MAPE after refit on train+val, scored on test)
        - val_metrics_by_model (RMSE/MAE/MAPE scored on validation; used for selection)
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

    # Metrics dicts
    val_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics: Dict[str, Dict[str, float]] = {}

    # Helper function to calculate all metrics
    def calculate_metrics(forecast_prices: np.ndarray, actual_prices: np.ndarray) -> Dict[str, float]:
        """Calculate RMSE, MAE, and MAPE."""
        rmse = float(np.sqrt(mean_squared_error(actual_prices, forecast_prices)))
        mae = float(mean_absolute_error(actual_prices, forecast_prices))
        mape = float(mean_absolute_percentage_error(actual_prices, forecast_prices) * 100)
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    # -----------------------
    # Validation step: train -> val (used for selection)
    # -----------------------
    # 1. Random Walk with Drift (baseline)
    try:
        drift = float(train_returns.mean())
        forecast_returns = np.full(forecast_periods, drift)
        start_price = float(train_prices.iloc[-1])
        forecast_prices = start_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = val_prices.iloc[:forecast_periods].values
        val_metrics["random_walk"] = calculate_metrics(forecast_prices, actual_prices)
    except Exception:
        pass

    # 2. AR(1) model
    # Note: AR(1) models mean-revert quickly and may perform poorly on long horizons
    try:
        # Ensure seed is set before AR(1) training
        np.random.seed(42)
        ar1_model = train_ar1(train_returns, train_exog)
        if train_exog is not None and not train_exog.empty:
            # Use last exogenous value from train set (not validation set) to avoid data leakage
            train_exog_array = np.vstack([train_exog.iloc[-1].values] * forecast_periods)
        else:
            train_exog_array = None
        forecast_returns = forecast_ar1(ar1_model, forecast_periods, train_exog_array)
        start_price = float(train_prices.iloc[-1])
        forecast_prices = start_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = val_prices.iloc[:forecast_periods].values
        metrics = calculate_metrics(forecast_prices, actual_prices)
        # Only include if MAPE is reasonable (<100% to avoid extreme failures)
        if metrics["MAPE"] < 100:
            val_metrics["ar1"] = metrics
    except Exception:
        pass

    # 3. ARIMA model
    try:
        # Ensure seed is set before ARIMA fitting
        np.random.seed(42)
        if model_type == "arima_fixed":
            if has_exog and train_exog is not None:
                arima_model = ARIMA(train_returns, order=(1, 0, 0), exog=train_exog).fit()
                # Use last exogenous value from train set (not validation set) to avoid data leakage
                train_exog_array = np.vstack([train_exog.iloc[-1].values] * forecast_periods)
                forecast_returns = arima_model.forecast(steps=forecast_periods, exog=train_exog_array).values
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
            if train_exog is not None and not train_exog.empty:
                # Use last exogenous value from train set (not validation set) to avoid data leakage
                train_exog_array = np.vstack([train_exog.iloc[-1].values] * forecast_periods)
                forecast_returns = forecast_from_arima(arima_model, "auto_arima", forecast_periods, train_exog_array)
            else:
                forecast_returns = forecast_from_arima(arima_model, "auto_arima", forecast_periods)

        start_price = float(train_prices.iloc[-1])
        forecast_prices = start_price * np.exp(np.cumsum(forecast_returns))
        actual_prices = val_prices.iloc[:forecast_periods].values
        val_metrics["arima"] = calculate_metrics(forecast_prices, actual_prices)
    except Exception:
        pass

    # 4. XGBoost (if enabled)
    # Note: XGBoost iterative forecasting accumulates errors on long horizons
    if use_ml:
        try:
            from .models import forecast_with_xgb, train_xgb_cv

            # Set seed before XGBoost training for reproducibility
            np.random.seed(42)
            xgb_model = train_xgb_cv(train_returns, train_exog)
            if xgb_model is not None:
                # Forecast using only train data (validation data should not be used for forecasting context in evaluation)
                forecast_returns = forecast_with_xgb(train_returns, train_exog, forecast_periods, model=xgb_model)
                if forecast_returns is not None:
                    start_price = float(train_prices.iloc[-1])
                    forecast_prices = start_price * np.exp(np.cumsum(forecast_returns))
                    actual_prices = val_prices.iloc[:forecast_periods].values
                    metrics = calculate_metrics(forecast_prices, actual_prices)
                    # Only include if MAPE is reasonable (<100% to avoid extreme failures on long horizons)
                    if metrics["MAPE"] < 100:
                        val_metrics["xgb"] = metrics
        except Exception:
            pass

    # 5. Regularized linear baselines (Ridge/Lasso/ElasticNet)
    try:
        ridge = train_linear_cv(train_returns, train_exog, model_kind="ridge")
        if ridge is not None:
            fr = forecast_with_linear(train_returns, train_exog, forecast_periods, model=ridge)
            if fr is not None:
                start_price = float(train_prices.iloc[-1])
                fp = start_price * np.exp(np.cumsum(fr))
                ap = val_prices.iloc[:forecast_periods].values
                val_metrics["ridge"] = calculate_metrics(fp, ap)
    except Exception:
        pass

    try:
        lasso = train_linear_cv(train_returns, train_exog, model_kind="lasso")
        if lasso is not None:
            fr = forecast_with_linear(train_returns, train_exog, forecast_periods, model=lasso)
            if fr is not None:
                start_price = float(train_prices.iloc[-1])
                fp = start_price * np.exp(np.cumsum(fr))
                ap = val_prices.iloc[:forecast_periods].values
                val_metrics["lasso"] = calculate_metrics(fp, ap)
    except Exception:
        pass

    try:
        enet = train_linear_cv(train_returns, train_exog, model_kind="elasticnet")
        if enet is not None:
            fr = forecast_with_linear(train_returns, train_exog, forecast_periods, model=enet)
            if fr is not None:
                start_price = float(train_prices.iloc[-1])
                fp = start_price * np.exp(np.cumsum(fr))
                ap = val_prices.iloc[:forecast_periods].values
                val_metrics["elasticnet"] = calculate_metrics(fp, ap)
    except Exception:
        pass

    if not val_metrics:
        return "random_walk", None, "unknown", {}, {}

    # Select best model based on validation RMSE (no test leakage)
    best_model = min(val_metrics, key=lambda k: val_metrics[k]["RMSE"])

    # -----------------------
    # Test step: refit on train+val -> score on test (final reportable metrics)
    # -----------------------
    combined_returns = pd.concat([train_returns, val_returns])
    combined_prices = pd.concat([train_prices, val_prices])
    combined_exog = None
    if has_exog and train_exog is not None and val_exog is not None:
        combined_exog = pd.concat([train_exog, val_exog])

    # For all models, compute test metrics after refit on train+val.
    # This is safe because we do not use test metrics for selection.
    # Forecast start price is the last validation price (end of train+val).
    start_price_test = float(val_prices.iloc[-1])
    actual_prices_test = test_prices.iloc[:forecast_periods].values

    # Random walk
    try:
        drift = float(combined_returns.mean())
        fr = np.full(forecast_periods, drift)
        fp = start_price_test * np.exp(np.cumsum(fr))
        test_metrics["random_walk"] = calculate_metrics(fp, actual_prices_test)
    except Exception:
        pass

    # AR(1)
    try:
        np.random.seed(42)
        ar1_model = train_ar1(combined_returns, combined_exog)
        if combined_exog is not None and not combined_exog.empty:
            exog_arr = np.vstack([combined_exog.iloc[-1].values] * forecast_periods)
        else:
            exog_arr = None
        fr = forecast_ar1(ar1_model, forecast_periods, exog_arr)
        fp = start_price_test * np.exp(np.cumsum(fr))
        m = calculate_metrics(fp, actual_prices_test)
        if m["MAPE"] < 100:
            test_metrics["ar1"] = m
    except Exception:
        pass

    # ARIMA
    try:
        np.random.seed(42)
        if model_type == "arima_fixed":
            if combined_exog is not None and not combined_exog.empty:
                arima_model = ARIMA(combined_returns, order=(1, 0, 0), exog=combined_exog).fit()
                exog_arr = np.vstack([combined_exog.iloc[-1].values] * forecast_periods)
                fr = arima_model.forecast(steps=forecast_periods, exog=exog_arr).values
            else:
                arima_model = ARIMA(combined_returns, order=(1, 0, 0)).fit()
                fr = arima_model.forecast(steps=forecast_periods).values
        else:
            arima_model = auto_arima(
                combined_returns,
                exogenous=combined_exog if (combined_exog is not None and not combined_exog.empty) else None,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False,
                random_state=42,
                n_jobs=1,
            )
            if combined_exog is not None and not combined_exog.empty:
                exog_arr = np.vstack([combined_exog.iloc[-1].values] * forecast_periods)
                fr = forecast_from_arima(arima_model, "auto_arima", forecast_periods, exog_arr)
            else:
                fr = forecast_from_arima(arima_model, "auto_arima", forecast_periods)

        fp = start_price_test * np.exp(np.cumsum(fr))
        test_metrics["arima"] = calculate_metrics(fp, actual_prices_test)
    except Exception:
        pass

    # XGBoost
    if use_ml:
        try:
            from .models import train_xgb_cv
            np.random.seed(42)
            xgb_model = train_xgb_cv(combined_returns, combined_exog)
            if xgb_model is not None:
                fr = forecast_with_xgb(combined_returns, combined_exog, forecast_periods, model=xgb_model)
                if fr is not None:
                    fp = start_price_test * np.exp(np.cumsum(fr))
                    m = calculate_metrics(fp, actual_prices_test)
                    if m["MAPE"] < 100:
                        test_metrics["xgb"] = m
        except Exception:
            pass

    # Linear models
    for kind in ("ridge", "lasso", "elasticnet"):
        try:
            mdl = train_linear_cv(combined_returns, combined_exog, model_kind=kind)  # type: ignore[arg-type]
            if mdl is None:
                continue
            fr = forecast_with_linear(combined_returns, combined_exog, forecast_periods, model=mdl)
            if fr is None:
                continue
            fp = start_price_test * np.exp(np.cumsum(fr))
            test_metrics[kind] = calculate_metrics(fp, actual_prices_test)
        except Exception:
            pass

    # Final report uses test metrics of the selected model when available; otherwise fall back to validation.
    best_test_mape = None
    if best_model in test_metrics:
        best_test_mape = float(test_metrics[best_model]["MAPE"])
    else:
        best_test_mape = float(val_metrics[best_model]["MAPE"])

    # Determine signal quality based on TEST MAPE (final)
    if best_test_mape is not None and best_test_mape < 5:
        signal_quality = "high"
    elif best_test_mape is not None and best_test_mape < 15:
        signal_quality = "medium"
    else:
        signal_quality = "low"

    return best_model, best_test_mape, signal_quality, test_metrics, val_metrics
