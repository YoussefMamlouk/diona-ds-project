"""
Plotting helpers for forecasts and volatility diagnostics.
"""
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_forecast(
    ticker: str,
    prices: pd.Series,
    raw_prices: pd.Series,
    forecast_artifacts: Dict[str, object],
    save: bool = False,
    horizon_suffix: str = "",
) -> Optional[str]:
    """Plot the forecast and optionally save to `results/`."""
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
        if forecast_length <= 10:
            lookback_days = 90
        elif forecast_length <= 3:
            lookback_days = 365
        elif forecast_length <= 6:
            lookback_days = 540
        else:
            lookback_days = 730

        start_date = last_date - pd.Timedelta(days=lookback_days)

        # Filter plot_prices (original daily data) by start date
        historical_prices = plot_prices[plot_prices.index >= start_date]

        if len(historical_prices) < 3:
            historical_prices = plot_prices
    else:
        historical_prices = prices
        last_date = prices.index[-1] if len(prices) > 0 else None
        forecast_end = forecast_series.index[-1] if len(forecast_series) > 0 else None
        start_date = prices.index[0] if len(prices) > 0 else None

    # Plot historical prices using original daily frequency
    plt.plot(historical_prices.index, historical_prices, label="Historical Prices", color="tab:blue", linewidth=1.5)

    # Connect last historical point to forecast for better visualization
    if plot_prices is not None and len(plot_prices) > 0:
        last_price = float(plot_prices.iloc[-1])
        last_date = plot_prices.index[-1]
    elif raw_prices is not None and len(raw_prices) > 0:
        last_price = float(raw_prices.iloc[-1])
        last_date = raw_prices.index[-1]
    else:
        last_price = float(prices.iloc[-1])
        last_date = prices.index[-1]

    # --- Daily volatility simulation for plotting ---
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
        """Simulate a daily path for daily forecasts by adding volatility to each day."""
        if forecast_dates is None or len(forecast_dates) == 0:
            return None

        np.random.seed(_stable_seed_from_text(seed_text))

        all_days = pd.bdate_range(start=start_dt + pd.offsets.BDay(), end=forecast_dates[-1])
        if len(all_days) == 0:
            return None

        current_price = float(start_price)
        out_dates = []
        out_prices = []

        if period_sigma_annualized_pct is not None and len(period_sigma_annualized_pct) > 0:
            avg_vol_ann = float(period_sigma_annualized_pct.mean())
        else:
            avg_vol_ann = 20.0

        sigma_daily = (avg_vol_ann / 100.0) / np.sqrt(252)
        target_price = float(forecast_prices.iloc[-1])
        target_lr_total = float(np.log(target_price / start_price))

        n_days = len(all_days)
        drift_daily = target_lr_total / n_days if n_days > 0 else 0.0
        daily_shocks = np.random.normal(loc=0.0, scale=sigma_daily, size=n_days)
        daily_returns = drift_daily + daily_shocks

        actual_sum = float(daily_returns.sum())
        if abs(actual_sum - target_lr_total) > 1e-10:
            correction = (target_lr_total - actual_sum) / n_days
            daily_returns = daily_returns + correction

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
        """Simulate a daily path that hits each period endpoint price while exhibiting daily volatility."""
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

            seg_days = pd.bdate_range(seg_start + pd.offsets.BDay(), seg_end)
            n = len(seg_days)
            if n <= 0:
                continue

            if period_log_returns is not None and len(period_log_returns) > i:
                target_lr = float(period_log_returns[i])
            else:
                prev_endpoint = float(start_price) if i == 0 else float(period_end_prices.iloc[i - 1])
                curr_endpoint = float(period_end_prices.iloc[i])
                if prev_endpoint <= 0 or curr_endpoint <= 0:
                    target_lr = 0.0
                else:
                    target_lr = float(np.log(curr_endpoint / prev_endpoint))

            if period_sigma_annualized_pct is not None and len(period_sigma_annualized_pct) > i:
                vol_ann = float(period_sigma_annualized_pct.iloc[i])
                sigma_daily = (vol_ann / 100.0) / np.sqrt(252)
            else:
                sigma_daily = 0.0

            shocks = np.random.normal(loc=0.0, scale=sigma_daily, size=n)
            drift_daily = target_lr / n
            correction = (target_lr - (drift_daily * n + float(shocks.sum()))) / n
            daily_logr = drift_daily + shocks + correction

            for d, r in zip(seg_days, daily_logr):
                current_price = current_price * float(np.exp(r))
                out_dates.append(d)
                out_prices.append(current_price)

        if not out_dates:
            return None
        return pd.Series(out_prices, index=pd.DatetimeIndex(out_dates), name="simulated_daily_path")

    try:
        infer = pd.infer_freq(forecast_series.index)
    except Exception:
        infer = None
    is_period_horizon = infer in ("MS", "W-FRI") or ("month" in horizon_suffix) or ("year" in horizon_suffix) or ("week" in horizon_suffix)

    returns_are_constant = False
    if forecast_returns is not None:
        forecast_returns_array = forecast_returns if isinstance(forecast_returns, np.ndarray) else np.array(forecast_returns)
        returns_are_constant = len(forecast_returns_array) > 0 and np.allclose(forecast_returns_array, forecast_returns_array[0], rtol=1e-10)

    is_daily_forecast = (infer == "B" or infer is None) and not is_period_horizon
    is_short_daily = is_daily_forecast and forecast_length <= 15
    should_simulate_daily = is_period_horizon or (is_daily_forecast and (returns_are_constant or is_short_daily))

    simulated_daily = None
    if should_simulate_daily:
        if is_daily_forecast and (returns_are_constant or is_short_daily):
            simulated_daily = _simulate_daily_path_for_daily_forecast(
                start_price=last_price,
                start_dt=last_date,
                forecast_dates=forecast_series.index,
                forecast_prices=forecast_series,
                period_sigma_annualized_pct=sigma_forecast,
                seed_text=f"{ticker}|{horizon_suffix}|plot_daily_sim",
            )
        else:
            simulated_daily = _simulate_daily_path_matching_period_endpoints(
                start_price=last_price,
                start_dt=last_date,
                period_end_dates=forecast_series.index,
                period_end_prices=forecast_series,
                period_log_returns=forecast_returns,
                period_sigma_annualized_pct=sigma_forecast,
                seed_text=f"{ticker}|{horizon_suffix}|plot_daily_sim",
            )

    extended_dates = [last_date] + list(forecast_series.index)
    extended_forecast = [last_price] + list(forecast_series.values)

    if simulated_daily is not None and len(simulated_daily) > 0:
        plt.plot(simulated_daily.index, simulated_daily.values, color="red", linewidth=1.6, label="Simulated Daily Path (volatility)")
        plt.plot(forecast_series.index, forecast_series.values, color="red", linestyle="None", label="_nolegend_")
        plt.plot([last_date, forecast_series.index[0]], [last_price, float(forecast_series.iloc[0])], color="red", linestyle=":", linewidth=1.0)
    else:
        plt.plot(extended_dates, extended_forecast, label="Forecasted Prices", color="red")

    if len(historical_prices) > 0:
        plot_start = historical_prices.index[0]
        plot_end = forecast_end if forecast_end is not None else historical_prices.index[-1]
        padding_days = max(7, forecast_length * 2)
        xlim_start = plot_start - pd.Timedelta(days=padding_days)
        xlim_end = plot_end + pd.Timedelta(days=padding_days)
        plt.xlim(xlim_start, xlim_end)

    plt.title(f"{ticker} Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")

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

    extended_mc_p10 = [last_price] + list(mc_p10.values)
    extended_mc_p90 = [last_price] + list(mc_p90.values)
    extended_mc_p50 = [last_price] + list(mc_p50.values)

    plt.fill_between(extended_dates, extended_mc_p10, extended_mc_p90, color="orange", alpha=0.2, label="MC 10-90%")
    plt.plot(extended_dates, extended_mc_p50, color="orange", linestyle=":", label="MC Median", marker='x')
    plt.legend()
    plt.grid()
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
    plt.close()
    return saved_path


def plot_volatility_forecast(
    ticker: str,
    log_returns: pd.Series,
    sigma_forecast: pd.Series,
    save: bool = False,
    horizon_suffix: str = "",
    raw_prices: Optional[pd.Series] = None,
    sigma_daily_forecast: Optional[pd.Series] = None,
    sigma_fitted: Optional[pd.Series] = None,
) -> Optional[str]:
    """Plot historical volatility vs GARCH forecast."""
    daily_prices: Optional[pd.Series] = None
    if raw_prices is not None and len(raw_prices) > 0:
        try:
            gaps = pd.Series(raw_prices.index).sort_values().diff().dropna()
            median_gap_days = float(gaps.dt.total_seconds().median() / 86400.0) if len(gaps) else 999.0
            if median_gap_days <= 3.5:
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
        try:
            freq = pd.infer_freq(log_returns.index)
            if freq and 'W' in freq:
                annualization_factor = np.sqrt(52)
            elif freq and 'M' in freq:
                annualization_factor = np.sqrt(12)
            else:
                annualization_factor = np.sqrt(252)
        except Exception:
            annualization_factor = np.sqrt(252)

        rolling_vol = log_returns.rolling(window=21).std() * annualization_factor * 100
        rolling_vol = rolling_vol.dropna()
    else:
        rolling_vol = pd.Series(dtype=float)

    plt.figure(figsize=(12, 6))

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

    if sigma_daily_forecast is not None and len(sigma_daily_forecast) > 0:
        sigma_daily_clean = sigma_daily_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        sigma_points = sigma_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        if (sigma_points is None or sigma_points.empty) and sigma_daily_clean is not None and not sigma_daily_clean.empty:
            try:
                sigma_points = sigma_daily_clean.reindex(sigma_forecast.index, method="ffill").dropna()
            except Exception:
                sigma_points = sigma_daily_clean.iloc[:: max(1, int(len(sigma_daily_clean) / max(1, len(sigma_forecast))))].copy()

        daily_color = "tomato"
        same_index = False
        try:
            if sigma_points is not None and not sigma_points.empty:
                same_index = sigma_points.index.equals(sigma_daily_clean.index)
        except Exception:
            same_index = False

        if same_index:
            plt.plot(
                sigma_daily_clean.index,
                sigma_daily_clean.values,
                label="GARCH Forecast (annualized)",
                color=daily_color,
                linewidth=2,
                alpha=0.85,
                zorder=3,
            )
        else:
            plt.plot(
                sigma_daily_clean.index,
                sigma_daily_clean.values,
                label="GARCH(1,1) Forecast (daily, annualized)",
                color=daily_color,
                linewidth=2,
                alpha=0.85,
                zorder=3,
            )
            if sigma_points is not None and len(sigma_points) > 1:
                plt.plot(
                    sigma_points.index,
                    sigma_points.values,
                    label="GARCH Forecast (horizon)",
                    color="darkred",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.8,
                    zorder=5,
                )
    else:
        sigma_points = sigma_forecast.replace([np.inf, -np.inf], np.nan).dropna()
        if sigma_points is not None and len(sigma_points) == 1:
            point_ts = sigma_points.index[0]
            point_val = float(sigma_points.iloc[0])
            span = pd.Timedelta(days=3)
            plt.plot(
                [point_ts - span, point_ts + span],
                [point_val, point_val],
                label="GARCH(1,1) Forecast (annualized)",
                color="tomato",
                linewidth=2,
                zorder=4,
            )
        else:
            plt.plot(
                sigma_points.index,
                sigma_points.values,
                label="GARCH(1,1) Forecast (annualized)",
                color="tomato",
                linewidth=2,
                zorder=4,
            )

    if not rolling_vol.empty:
        last_hist_date = rolling_vol.index[-1]
        plt.axvline(x=last_hist_date, color="gray", linestyle="--", alpha=0.5, label="Forecast Start")

    if not sigma_forecast.empty:
        forecast_start = sigma_forecast.index[0]
        forecast_end = sigma_forecast.index[-1]
        forecast_duration = (forecast_end - forecast_start).days

        if forecast_duration <= 0 and len(sigma_forecast) > 1:
            try:
                freq = pd.infer_freq(sigma_forecast.index)
                if freq and 'D' in freq:
                    forecast_duration = len(sigma_forecast)
                elif freq and 'W' in freq:
                    forecast_duration = len(sigma_forecast) * 7
                elif freq and 'M' in freq:
                    forecast_duration = len(sigma_forecast) * 30
                else:
                    forecast_duration = len(sigma_forecast) * 30
            except Exception:
                forecast_duration = len(sigma_forecast) * 30
        elif forecast_duration <= 0:
            forecast_duration = 10

        if forecast_duration <= 10:
            history_days = 90
        elif forecast_duration <= 30:
            history_days = 120
        elif forecast_duration <= 90:
            history_days = 180
        elif forecast_duration <= 180:
            history_days = 365
        else:
            history_days = 730

        xlim_start = forecast_start - pd.Timedelta(days=history_days)
        xlim_end = forecast_end + pd.Timedelta(days=max(7, forecast_duration // 4))

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
    plt.close()
    return saved_path


def plot_volatility_backtest(
    ticker: str,
    vol_backtest: Dict[str, object],
    save: bool = False,
    horizon_suffix: str = "",
) -> Optional[str]:
    """Plot backtest predicted vs realized volatility for the holdout window."""
    if not isinstance(vol_backtest, dict):
        return None
    series = vol_backtest.get("series", {})
    if not isinstance(series, dict):
        return None

    realized = series.get("realized_vol")
    garch = series.get("garch_vol")
    baseline_ewma = series.get("baseline_ewma_vol")
    if not isinstance(realized, pd.Series) or not isinstance(garch, pd.Series):
        return None
    if realized.empty or garch.empty:
        return None

    window = vol_backtest.get("rolling_window")
    window_label = f"H={int(window)}" if isinstance(window, int) else "H"
    plt.figure(figsize=(12, 6))
    plt.plot(realized.index, realized.values, label=f"Realized Forward Vol ({window_label})", color="black", linewidth=1.6)
    plt.plot(garch.index, garch.values, label="GARCH Forecast", color="red", linewidth=1.8)
    if isinstance(baseline_ewma, pd.Series) and not baseline_ewma.empty:
        plt.plot(
            baseline_ewma.index,
            baseline_ewma.values,
            label="Baseline",
            color="darkgray",
            linestyle=":",
            linewidth=1.4,
        )

    plt.title(f"{ticker} Volatility Backtest")
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
            filename = f"volatility_backtest_{ticker}_{horizon_suffix}_{timestamp}.png"
        else:
            filename = f"volatility_backtest_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
    plt.close()
    return saved_path
