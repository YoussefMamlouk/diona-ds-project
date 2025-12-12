import unittest

from src.evaluation import generate_forecast
from src.data_loader import compute_horizon_settings, load_series_for_horizon


class TestForecasting(unittest.TestCase):
	def test_generate_forecast_returns_expected_keys_with_sample_data(self):
		"""Sanity test: generate_forecast should return a dict with core outputs when using sample data."""
		hs = compute_horizon_settings(10, "day")
		data = load_series_for_horizon("TSLA", hs, use_sample_data=True)

		# call generate_forecast with the pieces it expects
		result = generate_forecast(
			"TSLA",
			data["prices"],
			data.get("raw_prices"),
			data["log_returns"],
			data["exog_df"],
			hs,
		)

		# Basic contract checks
		self.assertIsInstance(result, dict)
		for key in ("forecast_series", "mc_p50", "sigma_forecast", "last_price"):
			self.assertIn(key, result, f"Missing key: {key}")


if __name__ == "__main__":
	unittest.main()

