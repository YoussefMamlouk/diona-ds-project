"""
Exploratory Data Analysis (EDA) module for stock data.

This module provides comprehensive exploratory analysis including:
- Price trends and patterns
- Returns distribution and statistics
- Volatility analysis
- Volume analysis
- Correlation analysis
- Statistical summaries
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Optional, Dict
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Import data loading utilities
from src.data_loader import fetch_yfinance


def load_stock_data(
    ticker: str,
    period: str = "5y",
    use_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Load stock data for EDA analysis.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., '1y', '2y', '5y', '10y')
        use_cache: Whether to use cached data if available
        cache_only: If True, never download; rely on cached CSV only.
        
    Returns:
        DataFrame with stock data (Open, High, Low, Close, Volume, etc.)
    """
    data = fetch_yfinance(ticker, period, "1d", use_cache=use_cache, cache_only=cache_only)
    
    if data.empty:
        raise ValueError(f"No data available for ticker {ticker}")
    
    # Ensure we have required columns
    if "Adj Close" not in data.columns and "Close" in data.columns:
        data["Adj Close"] = data["Close"]
    
    return data


def compute_returns_and_volatility(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute various return and volatility metrics.
    
    Returns:
        Dictionary with:
        - prices: Adjusted close prices
        - returns: Simple returns (pct_change)
        - log_returns: Log returns
        - volatility_21d: 21-day rolling volatility (annualized)
        - volatility_252d: 252-day rolling volatility (annualized)
    """
    prices = data["Adj Close"]
    
    # Simple returns
    returns = prices.pct_change().dropna()
    
    # Log returns
    log_returns = np.log(prices).diff().dropna()
    
    # Rolling volatility (annualized)
    volatility_21d = log_returns.rolling(window=21).std() * np.sqrt(252) * 100
    volatility_252d = log_returns.rolling(window=252).std() * np.sqrt(252) * 100
    
    return {
        "prices": prices,
        "returns": returns,
        "log_returns": log_returns,
        "volatility_21d": volatility_21d,
        "volatility_252d": volatility_252d,
    }


def generate_eda_report(
    ticker: str,
    period: str = "5y",
    save: bool = True,
    cache_only: bool = False,
) -> Dict[str, str]:
    """
    Generate comprehensive EDA report with plots and statistics.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period to analyze
        save: Whether to save plots to results/ directory
        cache_only: If True, never download; rely on cached CSV only.
        
    Returns:
        Dictionary with paths to saved files
    """
    print(f"\n{'='*70}")
    print(f"  EXPLORATORY DATA ANALYSIS: {ticker}")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading data for {ticker}...")
    data = load_stock_data(ticker, period, use_cache=True, cache_only=cache_only)
    print(f"✓ Loaded {len(data)} trading days of data")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}\n")
    
    # Compute metrics
    metrics = compute_returns_and_volatility(data)
    prices = metrics["prices"]
    returns = metrics["returns"]
    log_returns = metrics["log_returns"]
    vol_21d = metrics["volatility_21d"]
    vol_252d = metrics["volatility_252d"]
    volume = data["Volume"]
    
    saved_paths = {}
    
    # 1. Price Trend Analysis
    print("Generating price trend analysis...")
    path1 = plot_price_trend(ticker, data, prices, save=save)
    if path1:
        saved_paths["price_trend"] = path1
    
    # 2. Returns Distribution
    print("Generating returns distribution analysis...")
    path2 = plot_returns_distribution(ticker, returns, log_returns, save=save)
    if path2:
        saved_paths["returns_distribution"] = path2
    
    # 3. Volatility Analysis
    print("Generating volatility analysis...")
    path3 = plot_volatility_analysis(ticker, prices, log_returns, vol_21d, vol_252d, save=save)
    if path3:
        saved_paths["volatility"] = path3
    
    # 4. Volume Analysis
    print("Generating volume analysis...")
    path4 = plot_volume_analysis(ticker, data, prices, volume, save=save)
    if path4:
        saved_paths["volume"] = path4
    
    # 5. Correlation Analysis
    print("Generating correlation analysis...")
    path5 = plot_correlation_analysis(ticker, data, returns, volume, save=save)
    if path5:
        saved_paths["correlation"] = path5
    
    # 6. Statistical Summary
    print("Generating statistical summary...")
    path6 = generate_statistical_summary(ticker, data, prices, returns, log_returns, vol_21d, save=save)
    if path6:
        saved_paths["statistics"] = path6
    
    # Print key insights
    print_insights(ticker, data, prices, returns, log_returns, vol_21d, volume)
    
    print(f"\n{'='*70}")
    print(f"  EDA COMPLETE: {len(saved_paths)} plots generated")
    print(f"{'='*70}\n")
    
    if saved_paths:
        print("Saved files:")
        for key, path in saved_paths.items():
            print(f"  - {key}: {os.path.basename(path)}")
        print()
    
    return saved_paths


def plot_price_trend(ticker: str, data: pd.DataFrame, prices: pd.Series, save: bool = True) -> Optional[str]:
    """Plot comprehensive price trend analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'{ticker} - Price Trend Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price with moving averages
    ax1 = axes[0]
    ax1.plot(prices.index, prices.values, label='Adjusted Close', linewidth=1.5, alpha=0.8)
    
    # Moving averages
    ma_20 = prices.rolling(window=20).mean()
    ma_50 = prices.rolling(window=50).mean()
    ma_200 = prices.rolling(window=200).mean()
    
    ax1.plot(ma_20.index, ma_20.values, label='MA(20)', linewidth=1, alpha=0.7)
    ax1.plot(ma_50.index, ma_50.values, label='MA(50)', linewidth=1, alpha=0.7)
    if len(prices) >= 200:
        ax1.plot(ma_200.index, ma_200.values, label='MA(200)', linewidth=1, alpha=0.7)
    
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Price Trend with Moving Averages', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 2. Daily price range (High - Low)
    ax2 = axes[1]
    price_range = data['High'] - data['Low']
    ax2.fill_between(data.index, 0, price_range.values, alpha=0.5, color='orange', label='Daily Range')
    ax2.plot(data.index, price_range.values, linewidth=0.5, color='darkorange')
    ax2.set_ylabel('Price Range ($)', fontsize=11)
    ax2.set_title('Daily Price Range (High - Low)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    
    # 3. Cumulative returns
    ax3 = axes[2]
    cumulative_returns = (1 + prices.pct_change().fillna(0)).cumprod() - 1
    ax3.plot(cumulative_returns.index, cumulative_returns.values * 100, 
             linewidth=1.5, color='green', label='Cumulative Returns')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Cumulative Returns (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Cumulative Returns Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_price_trend_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
        plt.close()
        return saved_path
    else:
        plt.show()
        return None


def plot_returns_distribution(ticker: str, returns: pd.Series, log_returns: pd.Series, save: bool = True) -> Optional[str]:
    """Plot returns distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ticker} - Returns Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of returns
    ax1 = axes[0, 0]
    ax1.hist(returns.values * 100, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
    ax1.axvline(returns.median() * 100, color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median()*100:.2f}%')
    ax1.set_xlabel('Daily Returns (%)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot (normality test)
    ax2 = axes[0, 1]
    stats.probplot(returns.values, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of returns by year
    ax3 = axes[1, 0]
    returns_by_year = returns.groupby(returns.index.year)
    years = sorted(returns_by_year.groups.keys())
    data_by_year = [returns_by_year.get_group(year).values * 100 for year in years]
    bp = ax3.boxplot(data_by_year, labels=years, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Daily Returns (%)', fontsize=11)
    ax3.set_title('Returns Distribution by Year', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time series of returns
    ax4 = axes[1, 1]
    ax4.plot(returns.index, returns.values * 100, linewidth=0.5, alpha=0.6, color='steelblue')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.fill_between(returns.index, 0, returns.values * 100, 
                     where=(returns.values >= 0), alpha=0.3, color='green', label='Positive')
    ax4.fill_between(returns.index, 0, returns.values * 100, 
                     where=(returns.values < 0), alpha=0.3, color='red', label='Negative')
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Daily Returns (%)', fontsize=11)
    ax4.set_title('Daily Returns Over Time', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_returns_distribution_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
        plt.close()
        return saved_path
    else:
        plt.show()
        return None


def plot_volatility_analysis(ticker: str, prices: pd.Series, log_returns: pd.Series, 
                            vol_21d: pd.Series, vol_252d: pd.Series, save: bool = True) -> Optional[str]:
    """Plot volatility analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ticker} - Volatility Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rolling volatility (21-day and 252-day)
    ax1 = axes[0, 0]
    ax1.plot(vol_21d.index, vol_21d.values, label='21-Day Volatility', linewidth=1.5, alpha=0.8)
    if len(vol_252d.dropna()) > 0:
        ax1.plot(vol_252d.index, vol_252d.values, label='252-Day Volatility', linewidth=1.5, alpha=0.8, color='orange')
    ax1.axhline(y=vol_21d.mean(), color='red', linestyle='--', linewidth=1, 
               label=f'Mean 21d: {vol_21d.mean():.1f}%', alpha=0.7)
    ax1.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax1.set_title('Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 2. Volatility clustering (scatter of returns vs lagged returns)
    ax2 = axes[0, 1]
    if len(log_returns) > 1:
        returns_today = log_returns.values[1:]
        returns_yesterday = log_returns.values[:-1]
        ax2.scatter(returns_yesterday * 100, returns_today * 100, alpha=0.3, s=10, color='steelblue')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('Returns(t-1) (%)', fontsize=11)
        ax2.set_ylabel('Returns(t) (%)', fontsize=11)
        ax2.set_title('Volatility Clustering (Returns vs Lagged Returns)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Volatility distribution
    ax3 = axes[1, 0]
    vol_clean = vol_21d.dropna()
    if len(vol_clean) > 0:
        ax3.hist(vol_clean.values, bins=30, density=True, alpha=0.7, color='coral', edgecolor='black')
        ax3.axvline(vol_clean.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {vol_clean.mean():.1f}%')
        ax3.axvline(vol_clean.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {vol_clean.median():.1f}%')
        ax3.set_xlabel('21-Day Volatility (%)', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Distribution of Rolling Volatility', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # 4. Volatility by year
    ax4 = axes[1, 1]
    vol_by_year = vol_21d.groupby(vol_21d.index.year).mean()
    years = vol_by_year.index
    colors = ['red' if v > vol_21d.mean() else 'green' for v in vol_by_year.values]
    ax4.bar(years, vol_by_year.values, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=vol_21d.mean(), color='black', linestyle='--', linewidth=1, 
               label=f'Overall Mean: {vol_21d.mean():.1f}%', alpha=0.7)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Average Volatility (%)', fontsize=11)
    ax4.set_title('Average Volatility by Year', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_volatility_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
        plt.close()
        return saved_path
    else:
        plt.show()
        return None


def plot_volume_analysis(ticker: str, data: pd.DataFrame, prices: pd.Series, 
                        volume: pd.Series, save: bool = True) -> Optional[str]:
    """Plot volume analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ticker} - Volume Analysis', fontsize=16, fontweight='bold')
    
    # 1. Volume over time
    ax1 = axes[0, 0]
    ax1.plot(volume.index, volume.values / 1e6, linewidth=0.8, alpha=0.7, color='steelblue')
    ma_volume_20 = volume.rolling(window=20).mean()
    ax1.plot(ma_volume_20.index, ma_volume_20.values / 1e6, 
             label='20-Day MA', linewidth=1.5, color='orange')
    ax1.set_ylabel('Volume (Millions)', fontsize=11)
    ax1.set_title('Trading Volume Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 2. Price vs Volume (scatter)
    ax2 = axes[0, 1]
    # Deterministic sampling for reproducibility/performance
    if len(prices) > 1000:
        sample_idx = np.linspace(0, len(prices) - 1, 1000, dtype=int)
        prices_sample = prices.iloc[sample_idx]
        volume_sample = volume.iloc[sample_idx]
    else:
        prices_sample = prices
        volume_sample = volume
    
    ax2.scatter(volume_sample.values / 1e6, prices_sample.values, alpha=0.4, s=20, color='steelblue')
    ax2.set_xlabel('Volume (Millions)', fontsize=11)
    ax2.set_ylabel('Price ($)', fontsize=11)
    ax2.set_title('Price vs Volume Relationship', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume distribution
    ax3 = axes[1, 0]
    ax3.hist(volume.values / 1e6, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(volume.mean() / 1e6, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {volume.mean()/1e6:.1f}M')
    ax3.axvline(volume.median() / 1e6, color='green', linestyle='--', linewidth=2, 
               label=f'Median: {volume.median()/1e6:.1f}M')
    ax3.set_xlabel('Volume (Millions)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Volume Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Volume by year (average)
    ax4 = axes[1, 1]
    volume_by_year = volume.groupby(volume.index.year).mean()
    years = volume_by_year.index
    ax4.bar(years, volume_by_year.values / 1e6, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axhline(y=volume.mean() / 1e6, color='red', linestyle='--', linewidth=1, 
               label=f'Overall Mean: {volume.mean()/1e6:.1f}M', alpha=0.7)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Average Volume (Millions)', fontsize=11)
    ax4.set_title('Average Volume by Year', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_volume_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
        plt.close()
        return saved_path
    else:
        plt.show()
        return None


def plot_correlation_analysis(ticker: str, data: pd.DataFrame, returns: pd.Series, 
                              volume: pd.Series, save: bool = True) -> Optional[str]:
    """Plot correlation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ticker} - Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for correlation
    df_corr = pd.DataFrame({
        'Returns': returns.values,
        'Volume': volume.reindex(returns.index).values,
        'Abs_Returns': np.abs(returns.values),
    })
    df_corr = df_corr.dropna()
    
    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    corr_matrix = df_corr.corr()
    im = ax1.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_yticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax1.set_yticklabels(corr_matrix.columns)
    ax1.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    plt.colorbar(im, ax=ax1)
    
    # 2. Returns vs Volume scatter
    ax2 = axes[0, 1]
    # Deterministic sampling for reproducibility/performance
    if len(df_corr) > 1000:
        sample_idx = np.linspace(0, len(df_corr) - 1, 1000, dtype=int)
        returns_sample = df_corr['Returns'].iloc[sample_idx]
        volume_sample = df_corr['Volume'].iloc[sample_idx]
    else:
        returns_sample = df_corr['Returns']
        volume_sample = df_corr['Volume']
    
    ax2.scatter(volume_sample.values / 1e6, returns_sample.values * 100, 
               alpha=0.4, s=20, color='steelblue')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Volume (Millions)', fontsize=11)
    ax2.set_ylabel('Returns (%)', fontsize=11)
    ax2.set_title('Returns vs Volume', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Autocorrelation of returns
    ax3 = axes[1, 0]
    # Use a subset for performance
    returns_subset = returns.iloc[-500:] if len(returns) > 500 else returns
    max_lags = min(20, len(returns_subset) - 1)
    if max_lags > 0:
        autocorr = [returns_subset.autocorr(lag=i) for i in range(1, max_lags + 1)]
        lags = range(1, len(autocorr) + 1)
        ax3.bar(lags, autocorr, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±0.1 threshold')
        ax3.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Lag (days)', fontsize=11)
        ax3.set_ylabel('Autocorrelation', fontsize=11)
        ax3.set_title('Returns Autocorrelation Function', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for autocorrelation', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Returns Autocorrelation Function', fontsize=12, fontweight='bold')
    
    # 4. Rolling correlation between returns and volume
    ax4 = axes[1, 1]
    if len(df_corr) > 60:
        rolling_corr = df_corr['Returns'].rolling(window=60).corr(df_corr['Volume'])
        ax4.plot(returns.index[len(returns) - len(rolling_corr):], rolling_corr.values, 
                linewidth=1.5, color='steelblue')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('60-Day Rolling Correlation', fontsize=11)
        ax4.set_title('Rolling Correlation: Returns vs Volume', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_correlation_{ticker}_{timestamp}.png"
        saved_path = os.path.join(results_dir, filename)
        plt.savefig(saved_path, bbox_inches="tight", dpi=150)
        plt.close()
        return saved_path
    else:
        plt.show()
        return None


def generate_statistical_summary(ticker: str, data: pd.DataFrame, prices: pd.Series, 
                                 returns: pd.Series, log_returns: pd.Series, 
                                 vol_21d: pd.Series, save: bool = True) -> Optional[str]:
    """Generate and save statistical summary."""
    summary = []
    summary.append("="*70)
    summary.append(f"  STATISTICAL SUMMARY: {ticker}")
    summary.append("="*70)
    summary.append("")
    
    # Basic price statistics
    summary.append("PRICE STATISTICS")
    summary.append("-"*70)
    summary.append(f"Current Price:          ${prices.iloc[-1]:.2f}")
    summary.append(f"Price Range:            ${prices.min():.2f} - ${prices.max():.2f}")
    summary.append(f"Mean Price:             ${prices.mean():.2f}")
    summary.append(f"Median Price:           ${prices.median():.2f}")
    summary.append(f"Std Dev:                ${prices.std():.2f}")
    summary.append("")
    
    # Returns statistics
    summary.append("RETURNS STATISTICS")
    summary.append("-"*70)
    summary.append(f"Mean Daily Return:      {returns.mean()*100:.4f}%")
    summary.append(f"Median Daily Return:    {returns.median()*100:.4f}%")
    summary.append(f"Std Dev (Daily):        {returns.std()*100:.4f}%")
    summary.append(f"Annualized Return:      {(returns.mean() * 252)*100:.2f}%")
    summary.append(f"Annualized Volatility:  {(returns.std() * np.sqrt(252))*100:.2f}%")
    summary.append(f"Skewness:               {returns.skew():.4f}")
    summary.append(f"Kurtosis:               {returns.kurtosis():.4f}")
    summary.append(f"Min Return:             {returns.min()*100:.2f}%")
    summary.append(f"Max Return:             {returns.max()*100:.2f}%")
    summary.append("")
    
    # Volatility statistics
    vol_clean = vol_21d.dropna()
    if len(vol_clean) > 0:
        summary.append("VOLATILITY STATISTICS (21-Day Rolling)")
        summary.append("-"*70)
        summary.append(f"Mean Volatility:       {vol_clean.mean():.2f}%")
        summary.append(f"Median Volatility:     {vol_clean.median():.2f}%")
        summary.append(f"Min Volatility:        {vol_clean.min():.2f}%")
        summary.append(f"Max Volatility:        {vol_clean.max():.2f}%")
        summary.append(f"Std Dev of Volatility: {vol_clean.std():.2f}%")
        summary.append("")
    
    # Normality test
    summary.append("NORMALITY TESTS")
    summary.append("-"*70)
    from scipy.stats import jarque_bera, shapiro
    jb_stat, jb_pvalue = jarque_bera(returns.values)
    summary.append(f"Jarque-Bera Test:")
    summary.append(f"  Statistic:            {jb_stat:.4f}")
    summary.append(f"  p-value:               {jb_pvalue:.4f}")
    summary.append(f"  Result:                {'Normal' if jb_pvalue > 0.05 else 'Not Normal'}")
    
    if len(returns) <= 5000:  # Shapiro-Wilk only for smaller samples
        sw_stat, sw_pvalue = shapiro(returns.values)
        summary.append(f"Shapiro-Wilk Test:")
        summary.append(f"  Statistic:            {sw_stat:.4f}")
        summary.append(f"  p-value:               {sw_pvalue:.4f}")
        summary.append(f"  Result:                {'Normal' if sw_pvalue > 0.05 else 'Not Normal'}")
    summary.append("")
    
    # Data quality
    summary.append("DATA QUALITY")
    summary.append("-"*70)
    summary.append(f"Total Trading Days:     {len(data)}")
    summary.append(f"Date Range:            {data.index[0].date()} to {data.index[-1].date()}")
    summary.append(f"Missing Values:         {data.isnull().sum().sum()}")
    summary.append(f"Zero Volume Days:      {(data['Volume'] == 0).sum()}")
    summary.append("")
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    if save:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"eda_statistics_{ticker}_{timestamp}.txt"
        saved_path = os.path.join(results_dir, filename)
        with open(saved_path, 'w') as f:
            f.write(summary_text)
        return saved_path
    return None


def print_insights(ticker: str, data: pd.DataFrame, prices: pd.Series, returns: pd.Series, 
                  log_returns: pd.Series, vol_21d: pd.Series, volume: pd.Series):
    """Print key insights from the analysis."""
    print("\n" + "="*70)
    print("  KEY INSIGHTS")
    print("="*70)
    
    # Price insights
    price_change = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
    print(f"\n📈 PRICE TRENDS:")
    print(f"  • Total return over period: {price_change:+.2f}%")
    print(f"  • Current price: ${prices.iloc[-1]:.2f}")
    print(f"  • Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Returns insights
    annual_return = returns.mean() * 252 * 100
    annual_vol = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    print(f"\n📊 RETURNS CHARACTERISTICS:")
    print(f"  • Annualized return: {annual_return:+.2f}%")
    print(f"  • Annualized volatility: {annual_vol:.2f}%")
    print(f"  • Sharpe ratio (approx): {sharpe_ratio:.2f}")
    print(f"  • Skewness: {returns.skew():.4f} {'(right-skewed)' if returns.skew() > 0 else '(left-skewed)' if returns.skew() < 0 else '(symmetric)'}")
    print(f"  • Kurtosis: {returns.kurtosis():.4f} {'(fat tails)' if returns.kurtosis() > 3 else '(normal tails)' if returns.kurtosis() < 3 else '(normal)'}")
    
    # Volatility insights
    vol_clean = vol_21d.dropna()
    if len(vol_clean) > 0:
        vol_trend = "increasing" if vol_clean.iloc[-30:].mean() > vol_clean.iloc[:30].mean() else "decreasing"
        print(f"\n📉 VOLATILITY PATTERNS:")
        print(f"  • Average volatility: {vol_clean.mean():.2f}%")
        print(f"  • Volatility range: {vol_clean.min():.2f}% - {vol_clean.max():.2f}%")
        print(f"  • Recent trend: {vol_trend}")
        if vol_clean.mean() > 30:
            print(f"  • ⚠ High volatility stock (volatility > 30%)")
        elif vol_clean.mean() > 20:
            print(f"  • Moderate volatility stock (volatility 20-30%)")
        else:
            print(f"  • Lower volatility stock (volatility < 20%)")
    
    # Volume insights
    avg_volume = volume.mean() / 1e6
    recent_volume = volume.iloc[-30:].mean() / 1e6
    volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
    
    print(f"\n📦 VOLUME ANALYSIS:")
    print(f"  • Average daily volume: {avg_volume:.1f}M shares")
    print(f"  • Recent average (30d): {recent_volume:.1f}M shares")
    print(f"  • Volume trend: {volume_trend}")
    
    # Risk assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    if annual_vol > 40:
        risk_level = "VERY HIGH"
    elif annual_vol > 30:
        risk_level = "HIGH"
    elif annual_vol > 20:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    print(f"  • Risk level: {risk_level} (based on {annual_vol:.1f}% volatility)")
    
    # Normality
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(returns.values)
    print(f"\n🔬 STATISTICAL PROPERTIES:")
    print(f"  • Returns distribution: {'Normal' if jb_pvalue > 0.05 else 'Non-normal'}")
    print(f"  • Jarque-Bera p-value: {jb_pvalue:.4f}")
    if jb_pvalue < 0.05:
        print(f"  • ⚠ Returns deviate from normal distribution (common in financial data)")
    
    print("\n" + "="*70 + "\n")



