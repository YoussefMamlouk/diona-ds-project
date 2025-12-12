import unittest
import pandas as pd
import numpy as np

class TestFetchers(unittest.TestCase):
    def test_fetch_yfinance_returns_df(self):
        # Monkeypatch yfinance.download indirectly by calling the fetcher and
        # ensuring it returns a DataFrame. We'll import and call directly.
        from src.lib.fetchers import fetch_yfinance
        # We don't mock network here; we just assert function is callable and returns a DataFrame type.
        df = fetch_yfinance('AAPL', '7d', '1d')
        self.assertIsNotNone(df)
        self.assertTrue(hasattr(df, 'columns'))

if __name__ == '__main__':
    unittest.main()
