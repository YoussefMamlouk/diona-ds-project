"""Risk analysis utilities.

Provide a small set of standard risk metrics (Value at Risk, expected shortfall)
that can be used by the rest of the project. These are intentionally lightweight
and deterministic to make testing and reporting easier.
"""
from typing import Tuple
import numpy as np


def historical_var(returns: np.ndarray, alpha: float = 0.05) -> float:
	"""Compute historical Value at Risk (VaR) at level `alpha`.

	`returns` are assumed to be numeric (percent or decimal as used by caller).
	"""
	if len(returns) == 0:
		return float('nan')
	return float(np.percentile(returns, 100 * alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
	"""Compute expected shortfall (CVaR) for level `alpha`.
	"""
	if len(returns) == 0:
		return float('nan')
	threshold = np.percentile(returns, 100 * alpha)
	tail = returns[returns <= threshold]
	if len(tail) == 0:
		return float('nan')
	return float(np.mean(tail))
