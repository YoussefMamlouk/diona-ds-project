import pandas as pd
import yfinance as yf


def fetch_yfinance(ticker: str, period: str, interval: str, session=None) -> pd.DataFrame:
    """
    Fetch stock data using yfinance library (fixed in v0.2.66+).
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        # Ensure Adj Close column exists
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]
        
        return data
        
    except Exception as e:
        print(f"[yfinance error for {ticker}]: {e}")
        return pd.DataFrame()
