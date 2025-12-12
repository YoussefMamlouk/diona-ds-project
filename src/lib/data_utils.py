"""
Data utilities for building exogenous features.
"""
import numpy as np
import pandas as pd
from typing import Optional


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
