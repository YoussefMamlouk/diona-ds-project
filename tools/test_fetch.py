import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import main as app_main
from lib.fetchers import fetch_yfinance
import pandas as pd

s = app_main.compute_horizon_settings(4, 'month')
print('settings:', s)

df = fetch_yfinance('TSLA', s['download_period'], s['interval'])
print('fetched rows (raw df):', len(df))

prices = None
if isinstance(df.columns, pd.MultiIndex):
    if ('Adj Close', 'TSLA') in df.columns:
        prices = df[('Adj Close', 'TSLA')].dropna()
    elif ('Close', 'TSLA') in df.columns:
        prices = df[('Close', 'TSLA')].dropna()
else:
    if 'Adj Close' in df.columns:
        prices = df['Adj Close'].dropna()
    elif 'Close' in df.columns:
        prices = df['Close'].dropna()

if prices is None:
    print('No price column found in fetched data')
else:
    print('price points (raw index):', len(prices))
    prices.index = pd.to_datetime(prices.index)
    # emulate resampling as in main
    if s['mode'] == 'W':
        prices_resampled = prices.resample('W-FRI').last().dropna()
    elif s['mode'] == 'M':
        prices_resampled = prices.resample('ME').last().dropna()
    else:
        prices_resampled = prices
    print('price points (after resample):', len(prices_resampled))
    print(prices_resampled.tail())
