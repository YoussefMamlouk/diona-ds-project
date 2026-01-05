# Raw Data Directory

## Data Sources

This project uses financial market data from **Yahoo Finance** via the `yfinance` Python library. Data is downloaded automatically when running the main script - no manual download or API keys are required.

## Data Download

### Automatic Download
- Data is fetched automatically when you run `python main.py` with a ticker symbol
- The `yfinance` library handles data retrieval and may cache data internally
- No data files need to be manually placed in this directory

### Offline Mode
The project runs in offline mode by default using cached data. Simply run:

```bash
python main.py
```

This uses cached data from this directory for reproducible results without requiring internet access.

## Data Structure

When data is downloaded, it includes:
- **Price data**: Open, High, Low, Close, Adjusted Close
- **Volume**: Trading volume
- **Date range**: Automatically determined based on the investment horizon

## Data Caching

The `yfinance` library may cache data internally. The project uses cached data by default for offline/deterministic mode. The library's internal caching reduces redundant downloads.

## Notes

- **No manual data download required**: Data is fetched automatically
- **Offline mode by default**: Uses cached data for offline use without internet
- **Reproducible**: Cached data ensures deterministic results
- **No API keys needed**: Yahoo Finance data is publicly available

## Data Caching

The project automatically caches downloaded data to this directory:
- Downloaded data is saved as `yfinance_cache_<TICKER>_<HASH>.csv`
- On subsequent runs, cached data is used automatically (no re-download)
- This ensures reproducibility and allows offline use
- Cache files are created automatically on first download

## Data Files

This directory contains cached Yahoo Finance data files. These are created automatically when data is first downloaded. The cache ensures:
- **Reproducibility**: Same data used across runs
- **Offline capability**: Works without internet after first download
- **Efficiency**: Avoids repeated API calls

To force a fresh download, delete the relevant cache file(s) from this directory.
