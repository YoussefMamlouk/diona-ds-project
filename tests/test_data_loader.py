import unittest
import pandas as pd
import numpy as np
import types

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # create a simple synthetic DataFrame mimicking yfinance output
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='B')
        adj = np.linspace(100, 120, len(dates))
        vol = np.linspace(1000, 2000, len(dates))
        # Simple flat columns (no MultiIndex)
        df = pd.DataFrame({'Adj Close': adj, 'Close': adj, 'Volume': vol}, index=dates)
        self.df = df
        # monkeypatch fetch_yfinance
        import src.lib.fetchers as fetchers
        self._orig_fetch = fetchers.fetch_yfinance
        fetchers.fetch_yfinance = lambda ticker, period, interval: self.df
        # Also patch src.data_loader reference if it was already imported elsewhere
        try:
            import src.data_loader as dl
            if hasattr(dl, 'fetch_yfinance'):
                self._orig_dl_fetch = dl.fetch_yfinance
                dl.fetch_yfinance = fetchers.fetch_yfinance
            else:
                self._orig_dl_fetch = None
        except Exception:
            self._orig_dl_fetch = None

    def tearDown(self):
        import src.lib.fetchers as fetchers
        fetchers.fetch_yfinance = self._orig_fetch
        # restore data_loader patched reference if present
        try:
            import src.data_loader as dl
            if getattr(self, '_orig_dl_fetch', None) is not None:
                dl.fetch_yfinance = self._orig_dl_fetch
        except Exception:
            pass

    def test_load_series_basic(self):
        from src.data_loader import compute_horizon_settings, load_series_for_horizon
        settings = compute_horizon_settings(1, 'month')
        out = load_series_for_horizon('FAKE', settings)
        self.assertIn('prices', out)
        self.assertIn('log_returns', out)
        self.assertFalse(out['prices'].empty)
        self.assertFalse(out['log_returns'].empty)

if __name__ == '__main__':
    unittest.main()
