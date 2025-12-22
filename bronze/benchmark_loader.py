"""
Bronze Layer: Benchmark Data Loader
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Fetch benchmark data (SPY, QQQ, VIX) from yfinance
- Enable comparison of portfolios against market indices
- Provide beta calculation baseline

Data Format:
- INPUT: yfinance API (same format as price data)
- OUTPUT: Parquet in Lakehouse format

Benchmarks:
- SPY: S&P 500 ETF (primary benchmark)
- QQQ: NASDAQ 100 ETF (tech benchmark)
- IWM: Russell 2000 ETF (small cap)
- DIA: Dow Jones Industrial Average ETF
- VIX: CBOE Volatility Index (indirectly via ^VIX)

Business Context:
- All portfolio strategies must beat SPY to be considered successful
- Beta calculation requires market returns (SPY)
- VIX for risk regime detection
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("✓ yfinance available")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("⚠️ yfinance not installed. Install with: pip install yfinance")


# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_DIR = BRONZE_DIR / 'benchmarks_lakehouse'

# Benchmark tickers
BENCHMARKS = {
    'SPY': {
        'name': 'S&P 500 ETF',
        'description': 'Primary benchmark for US large cap stocks',
        'category': 'broad_market'
    },
    'QQQ': {
        'name': 'NASDAQ 100 ETF',
        'description': 'Tech-heavy index benchmark',
        'category': 'growth'
    },
    'IWM': {
        'name': 'Russell 2000 ETF',
        'description': 'Small cap stocks benchmark',
        'category': 'small_cap'
    },
    'DIA': {
        'name': 'Dow Jones Industrial Average ETF',
        'description': 'Blue chip stocks benchmark',
        'category': 'value'
    },
    '^VIX': {
        'name': 'CBOE Volatility Index',
        'description': 'Market fear gauge',
        'category': 'volatility'
    },
    'GLD': {
        'name': 'Gold ETF',
        'description': 'Safe haven asset',
        'category': 'commodity'
    },
    'TLT': {
        'name': '20+ Year Treasury Bond ETF',
        'description': 'Long-term bonds benchmark',
        'category': 'bonds'
    }
}


# =============================================================================
# DATA FETCHER
# =============================================================================

def fetch_benchmark_data(
    tickers: List[str] = None,
    start_date: str = '2010-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Fetch benchmark data from yfinance
    
    Args:
        tickers: List of benchmark tickers (default: all benchmarks)
        start_date: Start date
        end_date: End date (default: today)
        
    Returns:
        DataFrame with benchmark OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    if tickers is None:
        tickers = list(BENCHMARKS.keys())
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}...")
            
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if len(df) == 0:
                logger.warning(f"No data for {ticker}")
                continue
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Normalize column names
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only needed columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Add ticker column
            # Clean ticker (remove ^ prefix for storage)
            clean_ticker = ticker.replace('^', '')
            df['ticker'] = clean_ticker
            
            # Add metadata
            benchmark_info = BENCHMARKS.get(ticker, {})
            df['benchmark_name'] = benchmark_info.get('name', ticker)
            df['category'] = benchmark_info.get('category', 'unknown')
            
            all_data.append(df)
            
            logger.info(f"✓ Fetched {len(df):,} rows for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No benchmark data was fetched")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure proper types
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['volume'] = combined_df['volume'].fillna(0).astype('int64')
    
    # Add fetched_at timestamp
    combined_df['fetched_at'] = datetime.now()
    
    return combined_df


def generate_sample_benchmark_data(
    tickers: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Generate sample benchmark data for testing
    
    Args:
        tickers: List of benchmark tickers
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with sample benchmark data
    """
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'IWM']
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    np.random.seed(42)
    
    all_data = []
    
    base_prices = {
        'SPY': 300,
        'QQQ': 200,
        'IWM': 150,
        'DIA': 250,
        'VIX': 20,
        'GLD': 150,
        'TLT': 130
    }
    
    for ticker in tickers:
        clean_ticker = ticker.replace('^', '')
        base_price = base_prices.get(clean_ticker, 100)
        
        # Generate random walk prices
        returns = np.random.normal(0.0003, 0.015, size=len(date_range))
        
        # VIX behaves differently (mean-reverting)
        if 'VIX' in ticker:
            returns = np.random.normal(0, 0.05, size=len(date_range))
            prices = base_price + np.cumsum(returns * 5)
            prices = np.clip(prices, 10, 80)  # VIX realistic bounds
        else:
            prices = base_price * np.cumprod(1 + returns)
        
        for i, date in enumerate(date_range):
            price = prices[i]
            daily_vol = abs(np.random.normal(0, 0.01)) * price
            
            all_data.append({
                'date': date,
                'ticker': clean_ticker,
                'open': round(price - daily_vol/2, 2),
                'high': round(price + daily_vol, 2),
                'low': round(price - daily_vol, 2),
                'close': round(price, 2),
                'volume': int(np.random.exponential(50000000)),
                'benchmark_name': BENCHMARKS.get(ticker, {}).get('name', ticker),
                'category': BENCHMARKS.get(ticker, {}).get('category', 'unknown'),
                'fetched_at': datetime.now()
            })
    
    df = pd.DataFrame(all_data)
    logger.info(f"Generated {len(df):,} sample benchmark data points")
    
    return df


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_benchmark_data(
    source: str = 'auto',
    tickers: List[str] = None,
    start_date: str = '2010-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Main function to load benchmark data
    
    Args:
        source: Data source ('yfinance', 'sample', 'auto')
        tickers: List of benchmark tickers
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with benchmark data
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: BENCHMARK DATA INGESTION")
    logger.info("=" * 70)
    
    if tickers is None:
        tickers = list(BENCHMARKS.keys())
    
    df = None
    
    # Try yfinance first
    if source in ['auto', 'yfinance'] and YFINANCE_AVAILABLE:
        try:
            logger.info("Fetching from yfinance...")
            df = fetch_benchmark_data(tickers=tickers, start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.warning(f"Failed to fetch from yfinance: {e}")
    
    # Fall back to sample data
    if df is None or len(df) == 0:
        logger.info("Using sample benchmark data")
        df = generate_sample_benchmark_data(tickers=tickers, start_date=start_date, end_date=end_date)
    
    # Log summary
    logger.info(f"\n✓ Loaded {len(df):,} data points")
    logger.info(f"✓ Benchmarks: {df['ticker'].nunique()}")
    logger.info(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Benchmark summary
    logger.info("\nBenchmark Summary:")
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        first_price = ticker_df.iloc[0]['close']
        last_price = ticker_df.iloc[-1]['close']
        total_return = (last_price / first_price - 1) * 100
        logger.info(f"  {ticker}: {len(ticker_df):,} days, "
                   f"${first_price:.2f} → ${last_price:.2f} ({total_return:+.1f}%)")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(f"BENCHMARK INGESTION COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("=" * 70)
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """Save benchmark data to Lakehouse format"""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


def register_in_universe(df: pd.DataFrame):
    """Register benchmark tickers in the universe"""
    from utils.ticker_universe import get_universe
    
    universe = get_universe()
    tickers = df['ticker'].unique().tolist()
    universe.register_source('benchmarks', tickers)
    
    logger.info(f"✓ Registered {len(tickers):,} benchmarks in universe")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(start_date: str = '2010-01-01', test: bool = False):
    """Main execution function"""
    logger.info("")
    logger.info("🚀 BRONZE LAYER: BENCHMARK DATA LOADER")
    logger.info("")
    
    try:
        if test:
            start_date = '2024-01-01'
            logger.info("Running in TEST mode (2024 data only)")
        
        # Load benchmark data
        df = load_benchmark_data(start_date=start_date)
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        # Register in universe
        register_in_universe(df)
        
        logger.info("")
        logger.info("✅ Benchmark data loading completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Benchmark loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv
    start_date = '2010-01-01'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--start='):
            start_date = arg.split('=')[1]
    
    exit(main(start_date=start_date, test=test_mode))
