import pandas as pd
import yfinance as yf
import requests
import time

# --- Create a global session with user-agent to avoid Yahoo 429 blocks ---
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def safe_yahoo_download(ticker: str, period: str, interval: str, retries: int = 3, yf_session=None):
    """Download price data from Yahoo with retry + user-agent, avoiding 429 blocks."""
    session_arg = yf_session if yf_session is not None else session
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                session=session_arg,
            )

            # If Yahoo sends empty response (429 / blocked)
            if df.empty:
                print(f"[Yahoo blocked] Retry {attempt+1}/{retries} for {ticker}...")
                time.sleep(1.5)
                continue

            return df

        except Exception as e:
            print(f"Download error for {ticker}: {e}")
            time.sleep(1)

    return pd.DataFrame()  # fallback empty


def fetch_yfinance(ticker: str, period: str, interval: str, session_object=None):
    """Unified fetcher handling Yahoo with robust retry + history fallback."""
    yf_session = session_object if session_object is not None else session
    df = safe_yahoo_download(ticker, period, interval, yf_session=yf_session)

    # If still empty -> try history() API
    if df.empty:
        try:
            ticker_obj = yf.Ticker(ticker, session=yf_session)
            df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

    # Final fallback to Stooq (daily/weekly/monthly only)
    if df.empty:
        df = fetch_stooq(ticker, interval)
        if not df.empty and "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

    return df


def fetch_stooq(ticker: str, interval: str):
    """Fallback using Stooq CSV (supports daily/weekly/monthly)."""
    interval_map = {"1d": "d", "1wk": "w", "1mo": "m"}
    if interval not in interval_map:
        return pd.DataFrame()

    suffix = ".us" if not ticker.lower().endswith(".us") else ""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}{suffix}&i={interval_map[interval]}"

    try:
        df = pd.read_csv(url)
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()
