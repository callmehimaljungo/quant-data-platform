"""
Fetch sector metadata từ yfinance cho tất cả tickers trong Bronze data.

Chạy: python bronze/fetch_all_sectors.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict

from config import BRONZE_DIR, METADATA_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")

BATCH_SIZE = 50  # Tickers per batch
DELAY_BETWEEN_BATCHES = 5  # Seconds


def get_all_tickers_from_bronze() -> List[str]:
    """Get unique tickers from Bronze price data."""
    bronze_file = BRONZE_DIR / 'all_stock_data.parquet'
    
    if not bronze_file.exists():
        bronze_file = BRONZE_DIR / 'prices.parquet'
    
    if not bronze_file.exists():
        raise FileNotFoundError("No Bronze price data found")
    
    # Only load ticker column
    df = pd.read_parquet(bronze_file, columns=['Ticker'])
    tickers = df['Ticker'].unique().tolist()
    
    logger.info(f"Found {len(tickers):,} unique tickers in Bronze data")
    return tickers


def load_existing_metadata() -> Dict[str, dict]:
    """Load existing metadata to avoid re-fetching."""
    existing = {}
    
    # Check ticker_metadata.parquet
    metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
    if metadata_file.exists():
        df = pd.read_parquet(metadata_file)
        for _, row in df.iterrows():
            existing[row['ticker']] = {
                'ticker': row['ticker'],
                'sector': row.get('sector', 'Unknown'),
                'industry': row.get('industry', 'Unknown')
            }
        logger.info(f"Loaded {len(existing)} existing metadata entries")
    
    return existing


def fetch_ticker_info(ticker: str) -> dict:
    """Fetch sector/industry for a single ticker from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'company_name': info.get('shortName', info.get('longName', '')),
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'company_name': '',
        }


def fetch_all_sectors(max_tickers: int = None):
    """
    Fetch sector data for all tickers.
    
    Args:
        max_tickers: Limit number of tickers to fetch (for testing)
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed")
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("FETCHING SECTOR METADATA FOR ALL TICKERS")
    logger.info("=" * 70)
    
    # Get all tickers
    all_tickers = get_all_tickers_from_bronze()
    
    # Load existing to skip
    existing = load_existing_metadata()
    
    # Filter out already fetched
    tickers_to_fetch = [t for t in all_tickers if t not in existing]
    logger.info(f"Tickers to fetch: {len(tickers_to_fetch)} (skipping {len(existing)} existing)")
    
    if max_tickers:
        tickers_to_fetch = tickers_to_fetch[:max_tickers]
        logger.info(f"Limited to first {max_tickers} tickers")
    
    # Fetch in batches
    results = list(existing.values())  # Start with existing
    total_batches = (len(tickers_to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(tickers_to_fetch), BATCH_SIZE):
        batch = tickers_to_fetch[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
        
        for ticker in batch:
            info = fetch_ticker_info(ticker)
            results.append(info)
            
            # Small delay between tickers to avoid rate limiting
            import random
            time.sleep(0.2 + random.random() * 0.2)  # 0.2 - 0.4 seconds
        
        # Delay to avoid rate limiting
        if batch_num < total_batches:
            time.sleep(DELAY_BETWEEN_BATCHES)
        
        # Save progress every 10 batches
        if batch_num % 10 == 0:
            df = pd.DataFrame(results)
            METADATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = METADATA_DIR / 'ticker_metadata.parquet'
            df.to_parquet(output_path, index=False)
            logger.info(f"  Progress saved: {len(df)} tickers")
    
    # Final save
    df = pd.DataFrame(results)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METADATA_DIR / 'ticker_metadata.parquet'
    df.to_parquet(output_path, index=False)
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("SECTOR METADATA FETCH COMPLETED")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Total tickers: {len(df)}")
    logger.info(f"Sectors found:")
    print(df['sector'].value_counts())
    logger.info("=" * 70)
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch sector metadata')
    parser.add_argument('--max', type=int, default=None, 
                        help='Max tickers to fetch (for testing)')
    args = parser.parse_args()
    
    try:
        fetch_all_sectors(max_tickers=args.max)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
