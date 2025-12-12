import unittest
from src.data_loader import compute_horizon_settings, load_series_for_horizon
from src import models
from src.evaluation import generate_forecast


class TestModelsPipeline(unittest.TestCase):
    def test_train_xgb_cv_returns_model(self):
        hs = compute_horizon_settings(10, 'day')
        data = load_series_for_horizon('TSLA', hs, use_sample_data=True)
        log_returns = data['log_returns']
        exog = data['exog_df']
        model = models.train_xgb_cv(log_returns, exog)
        # model should be either an XGBRegressor or None if insufficient data
        if model is not None:
            from xgboost import XGBRegressor
            self.assertIsInstance(model, XGBRegressor)

    def test_garch_forecast_length(self):
        hs = compute_horizon_settings(10, 'day')
        data = load_series_for_horizon('TSLA', hs, use_sample_data=True)
        artifacts = generate_forecast('TSLA', data['prices'], data['raw_prices'], data['log_returns'], data['exog_df'], hs, num_sims=50, use_ml=False)
        sigma = artifacts.get('sigma_forecast')
        self.assertEqual(len(sigma), hs['steps'])


if __name__ == '__main__':
    unittest.main()
