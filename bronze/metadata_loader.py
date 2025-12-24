"""
Bronze Layer: Stock Metadata Loader
Author: Quant Data Platform Team
Date: 2024-12-22

Purpose:
- Fetch stock metadata from yfinance (NO API key required)
- Save raw JSON responses to preserve original format
- Convert to Parquet for Lakehouse storage
- Register tickers in universe for intersection filtering

Data Format:
- INPUT: yfinance API (JSON responses)
- OUTPUT: Parquet in Lakehouse format + raw JSON backup

Business Context:
- Enriches price data with sector, industry, market cap
- Enables sector-based analysis and classification
- Supports market cap categorization (Large/Mid/Small)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, METADATA_DIR, LOG_FORMAT, GICS_SECTORS

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

OUTPUT_DIR = BRONZE_DIR / 'stock_metadata_lakehouse'
RAW_JSON_DIR = BRONZE_DIR / 'stock_metadata_raw'
CHECKPOINT_FILE = BRONZE_DIR / 'metadata_checkpoint.json'
BATCH_SIZE = 50  # tickers per batch (yfinance rate limit friendly)
SLEEP_BETWEEN_BATCHES = 2  # seconds


# =============================================================================
# CHECKPOINT FUNCTIONS (Resume capability)
# =============================================================================

def load_checkpoint() -> Dict[str, Any]:
    """
    Load checkpoint file to resume from previous progress
    
    Returns:
        Dictionary with 'fetched_tickers' list and 'results' list
    """
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"✓ Loaded checkpoint: {len(checkpoint.get('fetched_tickers', []))} tickers already fetched")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    return {'fetched_tickers': [], 'results': []}


def save_checkpoint(fetched_tickers: List[str], results: List[Dict]) -> None:
    """
    Save checkpoint with atomic write to prevent corruption
    
    Args:
        fetched_tickers: List of already fetched ticker symbols
        results: List of fetched metadata dictionaries
    """
    checkpoint = {
        'fetched_tickers': fetched_tickers,
        'results': results,
        'last_updated': datetime.now().isoformat()
    }
    
    # Atomic write: write to temp file, then rename
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, default=str)
        
        # Atomic rename (works on Windows too)
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        temp_file.rename(CHECKPOINT_FILE)
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_file.exists():
            temp_file.unlink()


def clear_checkpoint() -> None:
    """Clear checkpoint file after successful completion"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("✓ Checkpoint file cleared")


# =============================================================================
# YFINANCE FETCHER
# =============================================================================

def fetch_ticker_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a single ticker from yfinance
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with metadata or None if failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if valid response
        if not info or 'symbol' not in info:
            return None
        
        # Extract relevant fields
        return {
            'ticker': ticker,
            'company_name': info.get('longName') or info.get('shortName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'exchange': info.get('exchange'),
            'currency': info.get('currency'),
            'country': info.get('country'),
            'website': info.get('website'),
            'description': info.get('longBusinessSummary'),  # TEXT field
            'employees': info.get('fullTimeEmployees'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'pe_ratio': info.get('trailingPE'),
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'raw_json': json.dumps(info),  # Preserve original JSON
            'fetched_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {str(e)}")
        return None


def fetch_batch_info(tickers: List[str]) -> List[Dict]:
    """
    Fetch metadata for a batch of tickers
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        List of metadata dictionaries
    """
    results = []
    
    for ticker in tickers:
        info = fetch_ticker_info(ticker)
        if info:
            results.append(info)
    
    return results


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_available_tickers() -> List[str]:
    """
    Load list of tickers from existing price data
    
    Returns:
        List of ticker symbols
    """
    prices_file = BRONZE_DIR / 'all_stock_data.parquet'
    
    if not prices_file.exists():
        raise FileNotFoundError(f"Price data not found: {prices_file}")
    
    logger.info(f"Loading ticker list from {prices_file}")
    df = pd.read_parquet(prices_file, columns=['Ticker'])
    tickers = df['Ticker'].unique().tolist()
    
    logger.info(f"Found {len(tickers):,} unique tickers")
    return tickers


def load_stock_metadata(
    tickers: Optional[List[str]] = None,
    max_tickers: Optional[int] = None,
    save_raw_json: bool = True
) -> pd.DataFrame:
    """
    Main function to load stock metadata from yfinance
    
    Args:
        tickers: List of tickers to fetch (None = load from price data)
        max_tickers: Maximum number of tickers to fetch (for testing)
        save_raw_json: Whether to save raw JSON responses
        
    Returns:
        DataFrame with metadata
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: STOCK METADATA INGESTION")
    logger.info("=" * 70)
    
    # Get ticker list
    if tickers is None:
        tickers = load_available_tickers()
    
    if max_tickers:
        tickers = tickers[:max_tickers]
        logger.info(f"Limited to {max_tickers} tickers for testing")
    
    # Load checkpoint for resume capability
    checkpoint = load_checkpoint()
    already_fetched = set(checkpoint.get('fetched_tickers', []))
    all_results = checkpoint.get('results', [])
    
    # Filter out already fetched tickers
    remaining_tickers = [t for t in tickers if t not in already_fetched]
    
    if already_fetched:
        logger.info(f"Resuming from checkpoint: {len(already_fetched)} already fetched, {len(remaining_tickers)} remaining")
    
    # Process in batches
    total_batches = (len(remaining_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Processing {len(remaining_tickers):,} tickers in {total_batches} batches")
    
    fetched_tickers = list(already_fetched)
    
    for i in range(0, len(remaining_tickers), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_tickers = remaining_tickers[i:i + BATCH_SIZE]
        
        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch_tickers)} tickers")
        
        batch_results = fetch_batch_info(batch_tickers)
        all_results.extend(batch_results)
        fetched_tickers.extend(batch_tickers)
        
        # Save checkpoint after each batch (enable resume)
        save_checkpoint(fetched_tickers, all_results)
        
        # Progress update
        if batch_num % 10 == 0:
            total_fetched = len(already_fetched) + i + len(batch_tickers)
            success_rate = len(all_results) / total_fetched * 100
            logger.info(f"Progress: {total_fetched:,}/{len(tickers):,} "
                       f"({success_rate:.1f}% success) - Checkpoint saved")
        
        # Rate limiting
        if i + BATCH_SIZE < len(remaining_tickers):
            time.sleep(SLEEP_BETWEEN_BATCHES)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        logger.error("No metadata was fetched!")
        raise ValueError("Failed to fetch any metadata")
    
    # Add timestamp
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    # Log summary
    success_rate = len(df) / len(tickers) * 100
    logger.info(f"\n✓ Fetched metadata for {len(df):,} tickers ({success_rate:.1f}%)")
    logger.info(f"✓ Sectors: {df['sector'].nunique()} unique")
    logger.info(f"✓ Industries: {df['industry'].nunique()} unique")
    
    # Sector distribution
    logger.info("\nSector Distribution:")
    sector_counts = df['sector'].value_counts().head(10)
    for sector, count in sector_counts.items():
        logger.info(f"  {sector}: {count:,}")
    
    # Save raw JSON (if enabled)
    if save_raw_json:
        RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
        json_file = RAW_JSON_DIR / f'metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        raw_data = df.to_dict(orient='records')
        with open(json_file, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        logger.info(f"✓ Saved raw JSON to {json_file}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(f"METADATA INGESTION COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("=" * 70)
    
    return df


def save_to_lakehouse(df: pd.DataFrame) -> str:
    """
    Save metadata DataFrame to Lakehouse format
    
    Args:
        df: Metadata DataFrame
        
    Returns:
        Path to saved table
    """
    from utils.lakehouse_helper import pandas_to_lakehouse
    
    logger.info(f"Saving to Lakehouse: {OUTPUT_DIR}")
    path = pandas_to_lakehouse(df, OUTPUT_DIR, mode="overwrite")
    
    logger.info(f"✓ Saved to {path}")
    return path


def register_in_universe(df: pd.DataFrame):
    """
    Register fetched tickers in the ticker universe
    
    Args:
        df: Metadata DataFrame
    """
    from utils.ticker_universe import get_universe
    
    universe = get_universe()
    tickers = df['ticker'].unique().tolist()
    universe.register_source('metadata', tickers)
    
    logger.info(f"✓ Registered {len(tickers):,} tickers in universe")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(max_tickers: Optional[int] = None, test: bool = False):
    """
    Main execution function
    
    Args:
        max_tickers: Maximum tickers to fetch (for testing)
        test: If True, only fetch 10 tickers for testing
    """
    logger.info("")
    logger.info("🚀 BRONZE LAYER: STOCK METADATA LOADER")
    logger.info("")
    
    try:
        if test:
            max_tickers = 10
            logger.info("Running in TEST mode (10 tickers only)")
        
        # Load metadata
        df = load_stock_metadata(max_tickers=max_tickers)
        
        # Save to Lakehouse
        save_to_lakehouse(df)
        
        # Clear checkpoint after successful completion
        clear_checkpoint()
        
        # Register in universe
        register_in_universe(df)
        
        logger.info("")
        logger.info("✅ Stock metadata loading completed!")
        logger.info(f"✅ Output: {OUTPUT_DIR}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Metadata loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("")
        return 1


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    test_mode = '--test' in sys.argv
    max_tickers = None
    
    for arg in sys.argv[1:]:
        if arg.isdigit():
            max_tickers = int(arg)
    
    exit(main(max_tickers=max_tickers, test=test_mode))
