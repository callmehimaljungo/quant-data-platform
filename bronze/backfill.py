"""
Bronze Layer: Historical Backfill
Fetch historical data from yfinance for a date range and append to Bronze lakehouse.

Usage:
    python bronze/backfill.py --start 2024-12-21 --end 2025-01-02
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")


def get_existing_tickers() -> List[str]:
    """Get list of tickers from existing Bronze data."""
    prices_lakehouse = BRONZE_DIR / 'prices_lakehouse'
    
    if prices_lakehouse.exists():
        parquet_files = list(prices_lakehouse.glob('*.parquet'))
        if parquet_files:
            # Read first file to get tickers
            df = pd.read_parquet(parquet_files[0])
            ticker_col = 'ticker' if 'ticker' in df.columns else 'Ticker'
            if ticker_col in df.columns:
                return df[ticker_col].unique().tolist()
    
    # Fallback to all_stock_data.parquet
    all_data = BRONZE_DIR / 'all_stock_data.parquet'
    if all_data.exists():
        df = pd.read_parquet(all_data, columns=['Ticker'])
        return df['Ticker'].unique().tolist()
    
    logger.warning("No existing Bronze data found. Using default tickers.")
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']


def fetch_historical_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 50
) -> pd.DataFrame:
    """
    Fetch historical data using orchestrator with automatic fallback.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        batch_size: Number of tickers per batch (for orchestrator)
    
    Returns:
        DataFrame with OHLCV data
    """
    from bronze.orchestrator import DataOrchestrator
    
    orchestrator = DataOrchestrator()
    
    # Show available collectors
    available = orchestrator.get_available_collectors('prices')
    logger.info(f"Available price collectors: {available['prices']}")
    
    # Use orchestrator to fetch with automatic fallback
    df = orchestrator.collect_prices(tickers, start_date, end_date)
    
    if not df.empty:
        df['ingested_at'] = datetime.now()
    
    return df


def append_to_bronze(df: pd.DataFrame) -> Path:
    """Append data to Bronze lakehouse."""
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    output_dir = BRONZE_DIR / 'prices_lakehouse'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backfill_{timestamp}.parquet"
    output_path = output_dir / filename
    
    df.to_parquet(output_path, index=False)
    logger.info(f"[OK] Saved to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Bronze Layer Historical Backfill')
    parser.add_argument('--start', '-s', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                        help='Tickers per batch (default: 50)')
    args = parser.parse_args()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: HISTORICAL BACKFILL")
    logger.info("=" * 70)
    logger.info(f"  Start Date: {args.start}")
    logger.info(f"  End Date: {args.end}")
    logger.info("")
    
    try:
        # Get tickers from existing data
        tickers = get_existing_tickers()
        logger.info(f"[OK] Found {len(tickers):,} tickers in existing data")
        
        # Fetch historical data
        df = fetch_historical_data(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            batch_size=args.batch_size
        )
        
        if df.empty:
            logger.warning("[WARN] No data fetched")
            return 1
        
        # Append to Bronze
        output_path = append_to_bronze(df)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("[OK] BACKFILL COMPLETED")
        logger.info(f"  Rows: {len(df):,}")
        logger.info(f"  Tickers: {df['ticker'].nunique()}")
        logger.info(f"  Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"  Output: {output_path}")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Next step: Run Silver layer with --incremental flag")
        logger.info("  python silver/clean.py --incremental")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
