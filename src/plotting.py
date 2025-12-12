"""
Plotting helpers — produce and optionally save forecast charts into `results/`.
"""
from typing import Dict, Optional
import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_forecast_save(ticker: str, prices, raw_prices, forecast_artifacts: Dict[str, object], save: bool = False) -> Optional[str]:
    """Plot the forecast and optionally save to `results/`.

    Returns the path to the saved image if `save=True`, otherwise None.
    """
    forecast_series = forecast_artifacts["forecast_series"]
    hybrid_upper = forecast_artifacts["hybrid_upper"]
    hybrid_lower = forecast_artifacts["hybrid_lower"]
    mc_p10 = forecast_artifacts["mc_p10"]
    mc_p50 = forecast_artifacts["mc_p50"]
    mc_p90 = forecast_artifacts["mc_p90"]

    plt.figure(figsize=(10, 5))
    try:
        if raw_prices is not None and len(raw_prices) > len(prices):
            plt.plot(raw_prices.index, raw_prices, label="Historical Prices (raw)", color="tab:gray", alpha=0.4)
    except Exception:
        pass
    plt.plot(prices.index, prices, label="Historical Prices")
    
    # Connect last historical point to forecast for better visualization
    last_price = prices.iloc[-1]
    last_date = prices.index[-1]
    extended_dates = [last_date] + list(forecast_series.index)
    extended_forecast = [last_price] + list(forecast_series.values)
    
    plt.plot(extended_dates, extended_forecast, label=f"Forecasted Prices", color="red", marker='o', markersize=5)
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
        filename = f"forecast_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight")
    plt.show()
    return saved_path
