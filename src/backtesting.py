"""Backtesting utilities.

This module provides a thin, well-named API surface for backtesting so
other project modules can `from src import backtesting` or `from src.backtesting
import run_backtest` as required by the course repo structure.

It re-exports the existing implementation in `src.lib.model_utils` to avoid
duplicating logic while keeping a top-level `src` layout tidy.
"""
from typing import Tuple

from .lib.model_utils import run_backtest as _run_backtest_impl


def run_backtest(log_returns, exog_df, forecast_periods: int, model_type: str, prices) -> Tuple[str, float, str]:
	"""Run backtests and return (best_model_name, best_mape, signal_quality).

	This is a thin wrapper around `src.lib.model_utils.run_backtest` and exists
	so the project has a `src/backtesting.py` module with the expected API.
	"""
	return _run_backtest_impl(log_returns, exog_df, forecast_periods, model_type, prices)
