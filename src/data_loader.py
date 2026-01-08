"""
Data loading helpers.
"""
import numpy as np
import math
import os
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


def _get_cache_filename(ticker: str, period: str, interval: str) -> str:
    """Generate a cache filename based on ticker only (overwrites old cache)."""
    # Use a single cache file per ticker to always keep the latest data
    # This ensures we always have the most recent data and don't accumulate old files
    return f"yfinance_cache_{ticker}.csv"


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
            # Convert to timezone-naive if needed.
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            print(f"[Cache] Loaded {ticker} data from cache ({len(data)} rows)")
            return data
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}")
            return None
    return None


def _save_to_cache(ticker: str, period: str, interval: str, data: pd.DataFrame) -> None:
    """Save data to the cache directory."""
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, _get_cache_filename(ticker, period, interval))
    data.to_csv(cache_file)
    print(f"[Cache] Saved {ticker} data to cache ({len(data)} rows)")


def fetch_yfinance(
    ticker: str,
    period: str,
    interval: str,
    session=None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Load cached stock data and download once if missing.
    
    Args:
        ticker: Stock ticker symbol
        period: Download period when cache is missing (e.g., "5y").
        interval: Data interval ('1d' or '1mo').
        session: Unused (kept for compatibility).
        use_cache: Unused (kept for compatibility).
        cache_only: If True, do not download and return empty on cache miss.
        
    Returns:
        DataFrame with stock data
    """
    cached_data = _load_from_cache(ticker, period, interval)
    if cached_data is None:
        if cache_only:
            print(f"[Cache] No cached data found for {ticker}. Cache-only mode.")
            return pd.DataFrame()

        print(f"[Download] Cache missing for {ticker}. Downloading {period} daily data...")
        try:
            downloaded = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            print(f"[Download] Failed to download {ticker}: {exc}")
            return pd.DataFrame()

        if downloaded.empty:
            print(f"[Download] No data returned for {ticker}.")
            return pd.DataFrame()

        downloaded.index = pd.to_datetime(downloaded.index)
        if hasattr(downloaded.index, "tz") and downloaded.index.tz is not None:
            downloaded.index = downloaded.index.tz_localize(None)

        _save_to_cache(ticker, period, "1d", downloaded)
        cached_data = downloaded

    if interval == "1mo":
        resampled = cached_data.resample("MS").last()
        print(f"[Cache] Resampled cached data to {interval} interval ({len(resampled)} rows)")
        return resampled

    if interval != "1d":
        print(f"[Cache] Unsupported interval: {interval}. Returning empty DataFrame.")
        return pd.DataFrame()

    return cached_data


def build_exog(
    prices: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Build exogenous feature DataFrame for forecasting models.
    
    Args:
        prices: Price series
        volume: Volume series
    
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
	cache_only: bool = False,
) -> Dict[str, object]:
	"""Fetch price series and prepare exogenous features.

	`horizon_settings` should be the dict returned by `src.main.compute_horizon_settings`.
	Returns a dict with `prices`, `raw_prices`, `log_returns`, `exog_df`, `volume`, `horizon_settings`.
	"""
    # Load cached daily data and resample to the requested interval.
	mode = horizon_settings["mode"]
	invested_days = horizon_settings["invested_days"]
	data = fetch_yfinance(
		ticker,
		"cached",
		horizon_settings["interval"],
		use_cache=True,
		cache_only=cache_only,
	)

	if data.empty:
		return {"prices": pd.Series(dtype=float), "raw_prices": pd.Series(dtype=float), "log_returns": pd.Series(dtype=float), "exog_df": pd.DataFrame(), "volume": pd.Series(dtype=float), "horizon_settings": horizon_settings}

	# pick price and volume
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

	raw_prices.index = pd.to_datetime(raw_prices.index)

	prices = raw_prices.copy()
	mode = horizon_settings["mode"]
	if mode == "M":
		# Use 'MS' (month start) for pandas 2.2+ compatibility
		prices = prices.resample("MS").last().dropna()

	exog_df = build_exog(prices, volume)

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
	"""Derive forecasting settings (interval, steps, etc.) from user input.

	Copied from previous `src.main` implementation and exposed here so other
	modules can import and keep `main` minimal.
	"""
	if unit == "day":
		steps = max(1, int(math.ceil(value)))
		mode = "D"
		interval = "1d"
		freq = "B"
		invested_days = steps
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

	label_unit = unit if value == 1 else f"{unit}s"
	label = f"{value:g} {label_unit}"

	return {
		"steps": steps,
		"mode": mode,
		"interval": interval,
		"freq": freq,
		"invested_days": invested_days,
		"label": label,
	}
