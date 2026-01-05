"""
Incremental Pipeline - Realtime Price Updates (SIMPLIFIED & SAFE)

This pipeline updates live prices WITHOUT corrupting historical metrics.

Logic:
1. Load existing risk_metrics.parquet from cache (has all historical metrics)
2. Fetch latest prices from Yahoo Finance  
3. Update ONLY the 'last_price' column
4. Save back to realtime_metrics.parquet
5. Upload to R2

DOES NOT recalculate: Sharpe, Volatility, Sector, Max Drawdown, Beta, Alpha
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GOLD_DIR
from utils.r2_sync import upload_file_to_r2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_base_metrics() -> Optional[pd.DataFrame]:
    """
    Load validated historical metrics from Gold cache.
    
    Returns:
        DataFrame with all historical metrics (Sharpe, Sector, Vol, etc.)
    """
    cache_file = GOLD_DIR / 'cache' / 'risk_metrics.parquet'
    
    if not cache_file.exists():
        logger.error(f"✗ Base metrics not found: {cache_file}")
        return None
    
    try:
        df = pd.read_parquet(cache_file)
        logger.info(f"✓ Loaded base metrics: {len(df)} tickers")
        logger.info(f"  Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to load base metrics: {e}")
        return None


def fetch_latest_prices(tickers: list) -> Optional[pd.DataFrame]:
    """
    Download latest prices from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with columns: ticker, last_price
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("✗ yfinance not installed. Run: pip install yfinance")
        return None
    
    logger.info(f"[1/3] Fetching latest prices for {len(tickers)} tickers...")
    
    try:
        # Download just today's data
        data = yf.download(
            tickers=tickers,
            period='1d',
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            logger.warning("⚠ No data from Yahoo Finance (market closed?)")
            return None
        
        # Extract latest close prices
        rows = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker]
                
                if ticker_data.empty:
                    continue
                
                latest_close = ticker_data['Close'].iloc[-1]
                
                if pd.notna(latest_close):
                    rows.append({
                        'ticker': ticker,
                        'last_price': float(latest_close)
                    })
                
            except Exception as e:
                logger.debug(f"Skipped {ticker}: {e}")
                continue
        
        if not rows:
            logger.warning("⚠ No valid price data")
            return None
        
        df = pd.DataFrame(rows)
        logger.info(f"   ✓ Fetched {len(df)} prices")
        return df
        
    except Exception as e:
        logger.error(f"✗ Failed to download prices: {e}")
        return None


def update_prices(base_metrics: pd.DataFrame, 
                  live_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Update prices in base metrics.
    
    Strategy: Only update 'last_price' column, keep everything else.
    
    Args:
        base_metrics: Historical metrics from cache
        live_prices: Latest prices from Yahoo
        
    Returns:
        Updated DataFrame
    """
    logger.info("[2/3] Updating prices...")
    
    # Create a copy to avoid modifying original
    updated = base_metrics.copy()
    
    # Update last_price for tickers we have new data for
    for _, row in live_prices.iterrows():
        ticker = row['ticker']
        new_price = row['last_price']
        
        mask = updated['ticker'] == ticker
        if mask.any():
            updated.loc[mask, 'last_price'] = new_price
    
    # Add timestamp
    updated['updated_at'] = datetime.now().isoformat()
    
    logger.info(f"   ✓ Updated {len(live_prices)} prices")
    return updated


def save_and_upload(metrics: pd.DataFrame) -> bool:
    """
    Save metrics locally and upload to R2.
    
    Args:
        metrics: Updated metrics DataFrame
        
    Returns:
        True if successful
    """
    logger.info("[3/3] Saving and uploading...")
    
    # Save locally
    cache_dir = GOLD_DIR / 'cache'
    cache_dir.mkdir(exist_ok=True, parents=True)
    local_file = cache_dir / 'realtime_metrics.parquet'
    
    try:
        metrics.to_parquet(local_file, index=False)
        logger.info(f"   ✓ Saved to {local_file}")
    except Exception as e:
        logger.error(f"✗ Failed to save: {e}")
        return False
    
    # Upload to R2
    try:
        r2_key = 'processed/gold/cache/realtime_metrics.parquet'
        success = upload_file_to_r2(local_file, r2_key)
        if success:
            logger.info(f"   ✓ Uploaded to R2: {r2_key}")
            return True
        else:
            logger.warning("⚠ R2 upload failed (check credentials)")
            return False
    except Exception as e:
        logger.error(f"✗ R2 upload error: {e}")
        return False


def run_incremental_pipeline():
    """Run the complete incremental pipeline."""
    logger.info("=" * 70)
    logger.info(f"INCREMENTAL PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Step 1: Load base metrics
    base_metrics = load_base_metrics()
    if base_metrics is None:
        logger.error("✗ FAILED: Cannot proceed without base metrics")
        return False
    
    # Get ticker list
    tickers = base_metrics['ticker'].tolist()
    
    # Step 2: Fetch latest prices
    live_prices = fetch_latest_prices(tickers)
    if live_prices is None:
        logger.warning("⚠ No new price data, skipping update")
        return False
    
    # Step 3: Update prices
    updated = update_prices(base_metrics, live_prices)
    
    # Step 4: Save and upload
    success = save_and_upload(updated)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if success:
        logger.info(f"✓ OK | Duration: {duration:.1f}s")
        logger.info("=" * 70)
        return True
    else:
        logger.error(f"✗ FAILED | Duration: {duration:.1f}s")
        logger.info("=" * 70)
        return False


if __name__ == '__main__':
    try:
        success = run_incremental_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
