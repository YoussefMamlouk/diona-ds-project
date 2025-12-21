"""
Thin models module that exposes forecasting helpers.
This module wraps model functions used by the CLI so code can import
`src.models` instead of relying on `src.main` internals.

Notes:
- XGBoost is used for the ML pipeline. Import is performed lazily inside
  functions so importing this module does not fail in editors or test
  environments where `xgboost` may not be installed. When ML functions are
  invoked and `xgboost` is missing a clear ImportError with install
  instructions is raised.
"""
from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

if TYPE_CHECKING:
    from xgboost import XGBRegressor

def _get_xgb_regressor_class():
    """Lazily import and return the XGBRegressor class or raise a helpful error."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except Exception as e:
        raise ImportError(
            "XGBoost is required for XGB model functions but failed to import.\n"
            "To install, use conda:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate stock-forecast\n"
            f"Original import error: {e}"
        ) from e

def train_xgb_cv(log_returns: pd.Series, exog_df: Optional[pd.DataFrame], lags: int = 5, n_splits: int = 3):
    """Train XGBoost using simple time-series cross-validation and return best fitted model.
    Returns None when there's not enough data to train.
    """
    XGBCls = _get_xgb_regressor_class()
    aligned_exog = None
    if exog_df is not None and not exog_df.empty:
        aligned_exog = exog_df.reindex(log_returns.index)
    def make_design(series: pd.Series, exog: Optional[pd.DataFrame]):
        df = pd.DataFrame({"target": series})
        for i in range(1, lags + 1):
            df[f"lag_{i}"] = series.shift(i)
        if exog is not None:
            for col in exog.columns:
                df[col] = exog[col]
        df = df.dropna()
        y_local = df["target"]
        X_local = df.drop(columns="target")
        return X_local, y_local
    X, y = make_design(log_returns, aligned_exog)
    if len(y) < max(20, lags + 1):
        return None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = float("inf")
    best_model = None
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05},
        {"max_depth": 4, "learning_rate": 0.03},
    ]
    for params in param_grid:
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = XGBCls(
                n_estimators=300,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, preds))
        mean_cv = float(np.mean(cv_scores))
        if mean_cv < best_score:
            best_score = mean_cv
            # refit on full data
            best_model = XGBCls(
                n_estimators=300,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
            )
            best_model.fit(X, y)
    return best_model

def forecast_with_xgb(log_returns: pd.Series, exog_df: Optional[pd.DataFrame], steps: int, lags: int = 5, model=None) -> Optional[np.ndarray]:
    """Iterative one-step XGBoost forecast on log-returns using lag features (and optional exogenous).
    If `model` is None the function will train a default XGBRegressor on the available data.
    """
    XGBCls = _get_xgb_regressor_class()
    aligned_exog = None
    if exog_df is not None and not exog_df.empty:
        aligned_exog = exog_df.reindex(log_returns.index)
    def make_design(series: pd.Series, exog: Optional[pd.DataFrame]):
        df = pd.DataFrame({"target": series})
        for i in range(1, lags + 1):
            df[f"lag_{i}"] = series.shift(i)
        if exog is not None:
            for col in exog.columns:
                df[col] = exog[col]
        df = df.dropna()
        y_local = df["target"]
        X_local = df.drop(columns="target")
        return X_local, y_local
    X_train, y_train = make_design(log_returns, aligned_exog)
    if len(y_train) < max(20, lags + 1):
        return None
    if model is None:
        model = XGBCls(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train)
    history = list(log_returns.values)
    exog_last = aligned_exog.iloc[-1] if aligned_exog is not None else None
    feature_cols = list(X_train.columns)
    preds = []
    for _ in range(steps):
        row = {}
        for col in feature_cols:
            if col.startswith("lag_"):
                lag_idx = int(col.split("_")[1])
                row[col] = history[-lag_idx]
            elif exog_last is not None and col in exog_last:
                row[col] = exog_last[col]
        input_df = pd.DataFrame([row], columns=feature_cols)
        next_ret = float(model.predict(input_df)[0])
        preds.append(next_ret)
        history.append(next_ret)
    return np.asarray(preds)

def train_ar1(log_returns: pd.Series, exog_df: Optional[pd.DataFrame] = None):
    """Train an explicit AR(1) model (autoregressive model of order 1).
    
    This is equivalent to ARIMA(1,0,0) but provided as a separate function
    for clarity and explicit model comparison.
    
    Args:
        log_returns: Time series of log returns
        exog_df: Optional DataFrame of exogenous variables
        
    Returns:
        Fitted ARIMA model with order (1,0,0)
    """
    if exog_df is not None and not exog_df.empty:
        aligned_exog = exog_df.reindex(log_returns.index).dropna()
        aligned_returns = log_returns.reindex(aligned_exog.index).dropna()
        if len(aligned_returns) < 10:
            raise ValueError("Insufficient data after alignment for AR(1) model")
        model = ARIMA(aligned_returns, order=(1, 0, 0), exog=aligned_exog).fit()
    else:
        model = ARIMA(log_returns, order=(1, 0, 0)).fit()
    return model

def forecast_ar1(model, steps: int, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
    """Forecast using a fitted AR(1) model.
    
    Args:
        model: Fitted ARIMA model with order (1,0,0)
        steps: Number of steps to forecast
        exog_future: Optional 2D array with shape (steps, n_exog) for exogenous variables
        
    Returns:
        Array of forecasted log returns
    """
    if exog_future is not None:
        return model.forecast(steps=steps, exog=exog_future).to_numpy().flatten()
    return model.forecast(steps=steps).to_numpy().flatten()

def forecast_from_arima(model, model_type: str, steps: int, exog_future: Optional[np.ndarray] = None):
    """Forecast using either a statsmodels ARIMAResults or a pmdarima AutoARIMA.
    exog_future (optional): 2D array-like with shape (steps, n_exog) used when
    the fitted model was trained with exogenous regressors.
    """
    if model_type == "arima_fixed":
        if exog_future is not None:
            return model.forecast(steps=steps, exog=exog_future).to_numpy().flatten()
        return model.forecast(steps=steps).to_numpy().flatten()
    if exog_future is not None:
        return np.asarray(model.predict(n_periods=steps, exogenous=exog_future)).flatten()
    return np.asarray(model.predict(n_periods=steps)).flatten()
