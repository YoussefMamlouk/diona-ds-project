"""Preprocessing helpers.

This module exposes a small, stable API for preprocessing and fetching
time-series data. The heavy lifting is implemented in `src.data_loader`, so
we re-export the most useful functions here for consistency with the
recommended project structure.
"""
from typing import Dict, Optional

from .data_loader import load_series_for_horizon, compute_horizon_settings


def load_for_horizon(ticker: str, value: float, unit: str, fred_api_key: Optional[str] = None, extra_history_period: Optional[str] = None) -> Dict[str, object]:
	"""Convenience wrapper: compute horizon settings and load prepared series.

	Returns the same dict as `load_series_for_horizon`.
	"""
	horizon_settings = compute_horizon_settings(value, unit)
	return load_series_for_horizon(ticker, horizon_settings, fred_api_key=fred_api_key, extra_history_period=extra_history_period)
