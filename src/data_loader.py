"""
Data loading helpers.
"""
import numpy as np
import math
import pandas as pd
import yfinance as yf
from typing import Dict, Optional


def fetch_yfinance(ticker: str, period: str, interval: str, session=None) -> pd.DataFrame:
    """
    Fetch stock data using yfinance library (fixed in v0.2.66+).
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        # Ensure Adj Close column exists
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]
        
        return data
        
    except Exception as e:
        print(f"[yfinance error for {ticker}]: {e}")
        return pd.DataFrame()


def build_exog(
    prices: pd.Series,
    volume: pd.Series,
    download_period: str,
    interval: str,
    mode: str,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Build exogenous feature DataFrame for forecasting models.
    
    Args:
        prices: Price series
        volume: Volume series
        download_period: Download period string
        interval: Data interval
        mode: Resampling mode ('D', 'W', 'M')
        fred_api_key: Optional FRED API key for macro data
    
    Returns:
        DataFrame with exogenous features aligned to price index.
    """
    if prices.empty:
        return pd.DataFrame()
    
    exog = pd.DataFrame(index=prices.index)
    
    # Volume feature (normalized)
    if not volume.empty:
        vol_aligned = volume.reindex(prices.index).ffill().fillna(0)
        if vol_aligned.std() > 0:
            exog["volume_norm"] = (vol_aligned - vol_aligned.mean()) / vol_aligned.std()
        else:
            exog["volume_norm"] = 0.0
    
    # Simple momentum features
    if len(prices) > 5:
        returns = prices.pct_change().fillna(0)
        exog["momentum_5"] = returns.rolling(window=5, min_periods=1).mean()
    
    if len(prices) > 10:
        exog["momentum_10"] = prices.pct_change().rolling(window=10, min_periods=1).mean()
    
    # Clean up any infinities or NaNs
    exog = exog.replace([np.inf, -np.inf], 0).fillna(0)
    
    return exog


def load_series_for_horizon(ticker: str, horizon_settings: Dict[str, object], fred_api_key: Optional[str] = None, extra_history_period: Optional[str] = None, use_sample_data: bool = False) -> Dict[str, object]:
	"""Fetch price series and prepare exogenous features.

	`horizon_settings` should be the dict returned by `src.main.compute_horizon_settings`.
	Returns a dict with `prices`, `raw_prices`, `log_returns`, `exog_df`, `volume`, `horizon_settings`.
	"""
	# If requested, create a small deterministic synthetic dataset for offline/demo use
	if use_sample_data:
		# deterministic RNG for sample generation
		rs = np.random.RandomState(42)
		# generate 180 business days up to today
		dates = pd.bdate_range(end=pd.Timestamp.today(), periods=180)
		prices_arr = 100.0 + rs.normal(loc=0.0, scale=1.0, size=len(dates)).cumsum()
		raw_prices = pd.Series(data=prices_arr, index=dates)
		volume = pd.Series(data=(100000 + rs.randint(-5000, 5000, size=len(dates))), index=dates)
		data = None
	else:
		# Allow caller to override the computed download period to fetch extra history
		download_period = extra_history_period if extra_history_period is not None else horizon_settings["download_period"]
		data = fetch_yfinance(ticker, download_period, horizon_settings["interval"]) 

		if data.empty:
			return {"prices": pd.Series(dtype=float), "raw_prices": pd.Series(dtype=float), "log_returns": pd.Series(dtype=float), "exog_df": pd.DataFrame(), "volume": pd.Series(dtype=float), "horizon_settings": horizon_settings}

	# pick price and volume
	if not use_sample_data:
		if isinstance(data.columns, pd.MultiIndex):
			if ("Adj Close", ticker) in data.columns:
				raw_prices = data[("Adj Close", ticker)].dropna()
			elif ("Close", ticker) in data.columns:
				raw_prices = data[("Close", ticker)].dropna()
			else:
				# fallback to first numeric column
				cols = [c for c in data.columns if c[0] in ("Adj Close", "Close")]
				raw_prices = data[cols[0]].dropna() if cols else pd.Series(dtype=float)
			volume = data.get(("Volume", ticker), pd.Series(dtype=float)).dropna()
		else:
			if "Adj Close" in data.columns:
				raw_prices = data["Adj Close"].dropna()
			elif "Close" in data.columns:
				raw_prices = data["Close"].dropna()
			else:
				raw_prices = pd.Series(dtype=float)
			volume = data.get("Volume", pd.Series(dtype=float)).dropna()
	# if using sample data, `raw_prices` and `volume` were created earlier

	raw_prices.index = pd.to_datetime(raw_prices.index)

	prices = raw_prices.copy()
	mode = horizon_settings["mode"]
	if mode == "W":
		prices = prices.resample("W-FRI").last().dropna()
	elif mode == "M":
		# Use 'MS' (month start) for pandas 2.2+ compatibility
		prices = prices.resample("MS").last().dropna()

	exog_df = build_exog(prices, volume, horizon_settings["download_period"], horizon_settings["interval"], horizon_settings["mode"], fred_api_key)

	# log returns (aligned to resampled prices)
	if prices.empty:
		log_returns = pd.Series(dtype=float)
	else:
		log_returns = np.log(prices).diff().replace([np.inf, -np.inf], np.nan).dropna()

	exog_df = exog_df.reindex(log_returns.index).replace([np.inf, -np.inf], 0).fillna(0.0)

	# If we have raw daily prices, compute a rolling 21-day historical
	# volatility (annualized percent) and add it as an exogenous feature.
	try:
		if 'raw_prices' in locals() and raw_prices is not None and len(raw_prices) > 20:
			daily_logr = np.log(raw_prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
			rolling_std = daily_logr.rolling(window=21).std()
			rolling_vol = rolling_std * np.sqrt(252) * 100.0
			vol_aligned = rolling_vol.reindex(exog_df.index).ffill().fillna(0.0)
			exog_df["hist_vol"] = vol_aligned
	except Exception:
		pass

	return {
		"prices": prices,
		"raw_prices": raw_prices,
		"log_returns": log_returns,
		"exog_df": exog_df,
		"volume": volume,
		"horizon_settings": horizon_settings,
	}


def compute_horizon_settings(value: float, unit: str) -> Dict[str, object]:
	"""Derive forecasting settings (interval, download period, steps, etc.) from user input.

	Copied from previous `src.main` implementation and exposed here so other
	modules can import and keep `main` minimal.
	"""
	if unit == "day":
		steps = max(1, int(math.ceil(value)))
		mode = "D"
		interval = "1d"
		freq = "B"
		invested_days = steps
	elif unit == "week":
		steps = max(1, int(math.ceil(value)))
		mode = "W"
		interval = "1wk"
		freq = "W-FRI"
		invested_days = steps * 5
	elif unit == "month":
		steps = max(1, int(math.ceil(value)))
		mode = "M"
		interval = "1mo"
		freq = "MS"  # Month start - compatible with pandas 2.2+
		invested_days = steps * 21
	else:
		steps = max(1, int(math.ceil(value * 12)))
		mode = "M"
		interval = "1mo"
		freq = "MS"  # Month start - compatible with pandas 2.2+
		invested_days = steps * 21

	# Choose a download period that gives enough historical points for
	# resampling and backtesting. For monthly horizons prefer at least 1 year
	# of history so models have several monthly observations; for weekly prefer
	# at least 1 year as well. Daily horizons keep a shorter window.
	if mode == "D":
		download_days = int(max(math.ceil(invested_days * 1.2), 90))
	elif mode == "W":
		download_days = int(max(math.ceil(invested_days * 2), 252))
	else:  # monthly
		download_days = int(max(math.ceil(invested_days * 4), 365))

	if download_days <= 90:
		download_period = "90d"
	elif download_days <= 180:
		download_period = "6mo"
	elif download_days <= 252:
		download_period = "1y"
	elif download_days <= 504:
		download_period = "2y"
	elif download_days <= 1260:
		download_period = "5y"
	else:
		download_period = "10y"

	label_unit = unit if value == 1 else f"{unit}s"
	label = f"{value:g} {label_unit}"

	return {
		"steps": steps,
		"mode": mode,
		"interval": interval,
		"freq": freq,
		"invested_days": invested_days,
		"download_period": download_period,
		"label": label,
	}

