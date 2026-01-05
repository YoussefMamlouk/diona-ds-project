"""
Data loading helpers.
"""
import numpy as np
import math
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
import os


def _get_cache_filename(ticker: str, period: str, interval: str) -> str:
    """Generate a cache filename based on ticker only (overwrites old cache)."""
    # Use a single cache file per ticker to always keep the latest data
    # This ensures we always have the most recent data and don't accumulate old files
    return f"yfinance_cache_{ticker}.csv"


def _clean_old_cache_files(ticker: str):
    """Delete old cache files for this ticker, keeping only the latest one."""
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    if not os.path.exists(cache_dir):
        return
    
    # Find all cache files for this ticker (old hash-based naming)
    import glob
    old_pattern = os.path.join(cache_dir, f"yfinance_cache_{ticker}_*.csv")
    old_files = glob.glob(old_pattern)
    
    # Delete old hash-based cache files
    for old_file in old_files:
        try:
            os.remove(old_file)
        except Exception:
            pass


def _load_from_cache(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists."""
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, _get_cache_filename(ticker, period, interval))
    
    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            # Convert to timezone-naive if needed (yfinance returns timezone-aware)
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            print(f"[Cache] Loaded {ticker} data from cache ({len(data)} rows)")
            return data
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}")
            return None
    return None


def _save_to_cache(ticker: str, period: str, interval: str, data: pd.DataFrame):
    """Save downloaded data to cache, overwriting any old cache for this ticker."""
    if data.empty:
        return
    
    # Clean old cache files first
    _clean_old_cache_files(ticker)
    
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, _get_cache_filename(ticker, period, interval))
    
    try:
        # Make a copy to avoid modifying original
        data_to_save = data.copy()
        # Convert timezone-aware index to naive for CSV storage
        if data_to_save.index.tz is not None:
            data_to_save.index = data_to_save.index.tz_localize(None)
        data_to_save.to_csv(cache_file)
        print(f"[Cache] Saved {ticker} data to cache ({len(data)} rows, overwrote old cache)")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")


def fetch_yfinance(
    ticker: str,
    period: str,
    interval: str,
    session=None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance library with caching support.
    
    Note: For caching, we always fetch daily data (interval='1d') and cache it.
    The cache file is ticker-specific only, so it gets overwritten with the latest fetch.
    This ensures we always have the most recent and complete data.
    
    Args:
        ticker: Stock ticker symbol
        period: Download period (e.g., '1y', '2y', '3y')
        interval: Data interval (e.g., '1d', '1wk', '1mo') - but we always fetch '1d' for cache
        session: Optional yfinance session
        use_cache: If True, use cached data if available, and cache new downloads
        cache_only: If True, never download; return cached data or empty DataFrame.
        
    Returns:
        DataFrame with stock data
    """
    # Try to load from cache first (cache is always daily data)
    if use_cache or cache_only:
        # Cache uses simple filename: yfinance_cache_{ticker}.csv (always daily data)
        cached_data = _load_from_cache(ticker, period, interval)
        if cached_data is not None:
            # Cache always contains daily data, resample if needed
            if interval != "1d":
                # Resample cached daily data to requested interval
                if interval == "1wk":
                    resampled = cached_data.resample("W-FRI").last()
                elif interval == "1mo":
                    resampled = cached_data.resample("MS").last()
                else:
                    resampled = cached_data
                print(f"[Cache] Resampled cached data to {interval} interval ({len(resampled)} rows)")
                return resampled
            return cached_data

    # In cache-only mode, never attempt a download.
    if cache_only:
        print(f"[Cache] Cache-only mode enabled but no cache found for {ticker}.")
        return pd.DataFrame()
    
    # Download from Yahoo Finance
    # Always fetch daily data for maximum flexibility and caching
    fetch_interval = "1d"  # Always fetch daily, resample later if needed
    try:
        print(f"[Download] Fetching {ticker} data from Yahoo Finance (period={period}, interval={fetch_interval})...")
        ticker_obj = yf.Ticker(ticker)
        daily_data = ticker_obj.history(period=period, interval=fetch_interval)
        
        if daily_data.empty:
            print(f"[Download] No data returned for {ticker}")
            return pd.DataFrame()
        
        # Ensure Adj Close column exists
        if "Adj Close" not in daily_data.columns and "Close" in daily_data.columns:
            daily_data["Adj Close"] = daily_data["Close"]
        
        # Save to cache (always save daily data)
        if use_cache:
            _save_to_cache(ticker, period, "1d", daily_data)
        
        # Resample if needed (after saving daily to cache)
        # Note: We return resampled data but cache always contains daily data
        if interval != "1d":
            if interval == "1wk":
                data = daily_data.resample("W-FRI").last()
            elif interval == "1mo":
                data = daily_data.resample("MS").last()
            else:
                data = daily_data
            print(f"[Download] Successfully downloaded {len(data)} rows for {ticker} (resampled from {len(daily_data)} daily rows)")
        else:
            data = daily_data
            print(f"[Download] Successfully downloaded {len(data)} rows for {ticker}")
        
        return data
        
    except Exception as e:
        print(f"[Download] Error fetching {ticker} data: {e}")
        return pd.DataFrame()


def build_exog(
    prices: pd.Series,
    volume: pd.Series,
    download_period: str,
    interval: str,
    mode: str,
) -> pd.DataFrame:
    """Build exogenous feature DataFrame for forecasting models.
    
    Args:
        prices: Price series
        volume: Volume series
        download_period: Download period string
        interval: Data interval
        mode: Resampling mode ('D', 'W', 'M')
    
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


def load_series_for_horizon(
	ticker: str,
	horizon_settings: Dict[str, object],
	extra_history_period: Optional[str] = None,
	use_sample_data: bool = False,
	cache_only: bool = False,
) -> Dict[str, object]:
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
		# Always fetch daily data (most granular) and cache it
		# This allows resampling to any interval (daily, weekly, monthly) from the same cache
		# Calculate required download period based on horizon
		if extra_history_period is not None:
			download_period = extra_history_period
		else:
			# For caching: always fetch enough daily data, then resample as needed
			# This ensures we have enough data for any horizon
			mode = horizon_settings["mode"]
			invested_days = horizon_settings["invested_days"]
			if mode == "D":
				download_days = int(max(math.ceil(invested_days * 2.5), 365))
			elif mode == "W":
				download_days = int(max(math.ceil(invested_days * 3), 730))
			else:  # monthly
				download_days = int(max(math.ceil(invested_days * 5), 1095))
			
			# Map to yfinance period
			if download_days <= 90:
				download_period = "90d"
			elif download_days <= 180:
				download_period = "6mo"
			elif download_days <= 365:
				download_period = "1y"
			elif download_days <= 730:
				download_period = "2y"
			elif download_days <= 1095:
				download_period = "3y"
			elif download_days <= 1825:
				download_period = "5y"
			else:
				download_period = "10y"
		
		# Always fetch daily data for maximum flexibility
		# Pass the requested interval for display, but fetch daily and resample
		# The cache always stores daily data
		data = fetch_yfinance(
			ticker,
			download_period,
			horizon_settings["interval"],
			use_cache=True,
			cache_only=cache_only,
		)

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

	exog_df = build_exog(prices, volume, horizon_settings["download_period"], horizon_settings["interval"], horizon_settings["mode"])

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
	# resampling and backtesting. Fetch more data for medium/long-term forecasting.
	# For monthly horizons: at least 2 years of history (24+ monthly observations)
	# For weekly horizons: at least 2 years of history (104+ weekly observations)
	# For daily horizons: at least 1 year for short-term, more for longer horizons
	if mode == "D":
		# For daily: fetch at least 1 year, or 2x the forecast horizon (whichever is larger)
		download_days = int(max(math.ceil(invested_days * 2.5), 365))
	elif mode == "W":
		# For weekly: fetch at least 2 years (104 weeks) for proper backtesting
		download_days = int(max(math.ceil(invested_days * 3), 730))
	else:  # monthly
		# For monthly: fetch at least 3 years (36 months) for proper backtesting
		download_days = int(max(math.ceil(invested_days * 5), 1095))

	# Map download days to yfinance period strings
	# Always fetch enough data for proper model training and evaluation
	if download_days <= 90:
		download_period = "90d"
	elif download_days <= 180:
		download_period = "6mo"
	elif download_days <= 365:
		download_period = "1y"
	elif download_days <= 730:
		download_period = "2y"
	elif download_days <= 1095:
		download_period = "3y"
	elif download_days <= 1825:
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

