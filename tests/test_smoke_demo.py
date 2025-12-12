import unittest
from unittest.mock import patch, MagicMock


class SmokeDemoTest(unittest.TestCase):
    def test_main_demo_sample_mode_runs(self):
        # Patch argparse to simulate CLI args: demo + use_sample_data, no save
        fake_args = MagicMock()
        fake_args.demo = True
        fake_args.save_plot = False
        fake_args.use_sample_data = True
        fake_args.extra_history = None

        with patch('src.main.argparse.ArgumentParser.parse_known_args', return_value=(fake_args, [])):
            # Patch plotting to avoid writing files
            with patch('src.evaluation.plot_forecast', return_value=None):
                # Patch yfinance current price fetch to return a small dataframe
                class DummyHist:
                    def __init__(self):
                        import pandas as pd
                        self.df = pd.DataFrame({'Close': [100.0]}, index=[pd.Timestamp.today()])

                    def history(self, period='1d'):
                        return self.df

                with patch('yfinance.Ticker', return_value=DummyHist()):
                    # Import and run main (should not raise)
                    from src import main as app_main
                    app_main.main()


if __name__ == '__main__':
    unittest.main()
