# src/lib package
"""Library modules for stock forecasting tool."""

from .fetchers import fetch_yfinance
from .data_utils import build_exog
from .model_utils import run_backtest

__all__ = ["fetch_yfinance", "build_exog", "run_backtest"]
