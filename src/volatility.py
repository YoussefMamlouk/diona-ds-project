"""
Volatility backtesting utilities.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _select_vol_backtest_window(horizon_days: int) -> int:
    """Select a realized-vol window that matches the forecast horizon scale."""
    if horizon_days <= 15:
        return 10
    if horizon_days <= 35:
        return 21
    if horizon_days <= 80:
        return 63
    if horizon_days <= 160:
        return 126
    return 252


def backtest_garch_volatility(
    daily_log_returns: pd.Series,
    horizon_days: int,
    rolling_window: Optional[int] = None,
    max_backtest_days: Optional[int] = 252,
) -> Dict[str, object]:
    """Professional volatility backtest using rolling H-step forecasts.

    Compares GARCH forecasts against forward realized variance and strong baselines.
    Metrics include QLIKE (variance loss) and RMSE/MAE in annualized volatility units.
    Defaults to evaluating the most recent ~1 year of usable points to reduce regime noise.
    """
    if daily_log_returns is None or daily_log_returns.empty:
        return {}

    returns_pct = (100.0 * daily_log_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if returns_pct.empty:
        return {}

    horizon_days = int(max(1, horizon_days))
    if rolling_window is None:
        rolling_window = _select_vol_backtest_window(horizon_days)
    rolling_window = int(max(5, rolling_window))

    # Forward realized variance over the next H days (pct^2).
    sq_returns = returns_pct.pow(2)
    realized_forward_var = sq_returns.rolling(window=rolling_window).sum().shift(-rolling_window + 1)
    realized_forward_var = realized_forward_var.replace([np.inf, -np.inf], np.nan)
    valid_index = realized_forward_var.dropna().index
    if len(valid_index) < 30:
        return {}

    backtest_points = len(valid_index)
    if max_backtest_days is not None:
        backtest_points = min(backtest_points, int(max_backtest_days))
    test_index = valid_index[-backtest_points:]

    min_train = max(252, rolling_window * 3)
    if len(returns_pct) <= min_train:
        return {}

    try:
        from arch import arch_model
    except Exception:
        return {}

    # Precompute baseline (no lookahead): EWMA variance.
    lambda_ = 0.94
    ewma_vals = np.empty(len(returns_pct), dtype=float)
    init_var = float(sq_returns.iloc[:min(252, len(sq_returns))].mean()) if len(sq_returns) > 1 else 0.0
    ewma_vals[0] = init_var
    for i in range(1, len(returns_pct)):
        r2 = float(sq_returns.iloc[i - 1])
        ewma_vals[i] = lambda_ * ewma_vals[i - 1] + (1.0 - lambda_) * r2
    ewma_var = pd.Series(ewma_vals, index=returns_pct.index)

    pred_var_values = []
    pred_index = []
    for idx in test_index:
        pos = returns_pct.index.get_loc(idx)
        if pos <= min_train:
            continue
        train_slice = returns_pct.iloc[:pos]
        if len(train_slice) <= rolling_window:
            continue
        try:
            garch = arch_model(train_slice, vol="GARCH", p=1, o=1, q=1, dist="t")
            garch_fit = garch.fit(disp="off", options={'maxiter': 1000, 'disp': False})
        except Exception:
            try:
                garch = arch_model(train_slice, vol="GARCH", p=1, o=0, q=1, dist="normal")
                garch_fit = garch.fit(disp="off", options={'maxiter': 1000, 'disp': False})
            except Exception:
                continue

        scale = 1.0
        try:
            unconditional_var = float(garch_fit.unconditional_variance)
        except Exception:
            unconditional_var = float(train_slice.var())
        try:
            train_trailing_var = train_slice.pow(2).rolling(window=rolling_window).sum().dropna()
            target_mean_var = float(train_trailing_var.mean()) if not train_trailing_var.empty else float("nan")
            if np.isfinite(target_mean_var) and np.isfinite(unconditional_var) and unconditional_var > 0:
                scale = float(np.sqrt(max(target_mean_var / (rolling_window * unconditional_var), 0.0)))
        except Exception:
            scale = 1.0

        try:
            var_data = garch_fit.forecast(horizon=rolling_window).variance
            if hasattr(var_data, 'values'):
                var_arr = var_data.values
            elif hasattr(var_data, 'iloc'):
                var_arr = var_data.iloc[:, :].values
            else:
                var_arr = np.asarray(var_data)
            if var_arr.ndim == 2:
                var_path = var_arr[0, :]
            else:
                var_path = var_arr.flatten()
            if var_path.size == 0:
                continue
            var_path = np.asarray(var_path, dtype=float)
            var_path = np.maximum(var_path, 0.0)
            sum_var_pct2 = float(np.nansum(var_path))
            pred_var = float(max(sum_var_pct2 * (scale ** 2), 0.0))
        except Exception:
            continue
        pred_var_values.append(pred_var)
        pred_index.append(idx)

    garch_var = pd.Series(pred_var_values, index=pd.Index(pred_index)).replace([np.inf, -np.inf], np.nan).dropna()
    realized_var = realized_forward_var.loc[garch_var.index].replace([np.inf, -np.inf], np.nan).dropna()
    ewma_horizon_var = (ewma_var.shift(1) * rolling_window).loc[garch_var.index].replace([np.inf, -np.inf], np.nan).dropna()
    if garch_var.empty or realized_var.empty:
        return {}

    def _score(pred_var: pd.Series, true_var: pd.Series) -> Dict[str, float]:
        common = pred_var.index.intersection(true_var.index)
        if len(common) < 5:
            return {}
        pv = pred_var.loc[common].astype(float).clip(lower=1e-12)
        tv = true_var.loc[common].astype(float).clip(lower=0.0)
        qlike = float(np.mean(np.log(pv) + (tv / pv)))
        mse_var = float(np.mean((tv - pv) ** 2))
        vol_scale = float(np.sqrt(252.0 / rolling_window))
        rmse_vol = float(np.sqrt(np.mean((np.sqrt(tv) - np.sqrt(pv)) ** 2)) * vol_scale)
        mae_vol = float(np.mean(np.abs(np.sqrt(tv) - np.sqrt(pv))) * vol_scale)
        return {"RMSE": rmse_vol, "MAE": mae_vol, "QLIKE": qlike, "MSE_VAR": mse_var}

    garch_metrics = _score(garch_var, realized_var)
    ewma_metrics = _score(ewma_horizon_var, realized_var)

    common_index = garch_var.index.intersection(realized_var.index)
    points = int(len(common_index))
    vol_scale = float(np.sqrt(252.0 / rolling_window))
    realized_vol = np.sqrt(np.maximum(realized_var, 0.0)) * vol_scale
    garch_vol = np.sqrt(np.maximum(garch_var, 0.0)) * vol_scale
    ewma_vol = np.sqrt(np.maximum(ewma_horizon_var, 0.0)) * vol_scale

    return {
        "rolling_window": int(rolling_window),
        "points": points,
        "metrics": {
            "garch": garch_metrics,
            "baseline_ewma": ewma_metrics,
        },
        "series": {
            "realized_vol": realized_vol,
            "garch_vol": garch_vol,
            "baseline_ewma_vol": ewma_vol,
        },
    }
