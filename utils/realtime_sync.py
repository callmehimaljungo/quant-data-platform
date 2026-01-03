"""
Real-time Data Sync Utility

Handles continuous ingestion of stock prices (yfinance) and market news
for existing tickers in the universe. Syncs updates to R2.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Project imports
import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BRONZE_DIR, 
    REALTIME_PRICE_INTERVAL, 
    REALTIME_NEWS_INTERVAL,
    REALTIME_MAX_ITERATIONS
)
from utils.ticker_universe import get_universe
from utils.r2_sync import upload_bronze_to_r2

# Try importing yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('realtime_sync')


def fetch_latest_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest prices for tickers using yfinance with batching.
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not installed.")
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

    logger.info(f"Fetching prices for {len(tickers)} tickers (Batch processing)...")
    
    # Batch processing to prevent timeouts and blocking
    BATCH_SIZE = 50
    all_records = []
    
    # Valid tickers cache (could be persisted, but simple set for now)
    # We proceed in chunks
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        try:
            # logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} tickers)")
            
            # Use threads=False for safer execution in loops, or True for speed
            data = yf.download(
                batch, 
                period="1d", 
                interval="5m", # 5m is good for realtime
                group_by='ticker',
                progress=False,
                threads=True
                # show_errors=False removed (not supported in this version)
            )
            
            if data.empty:
                continue

            # Process Batch
            if len(batch) == 1:
                # Single ticker structure
                t = batch[0]
                df = data
                if not df.empty:
                    # Get last row
                    row = df.iloc[-1]
                    all_records.append({
                        'date': row.name,
                        'ticker': t,
                        'open': float(row.get('Open', 0)),
                        'high': float(row.get('High', 0)),
                        'low': float(row.get('Low', 0)),
                        'close': float(row.get('Close', 0)),
                        'volume': int(row.get('Volume', 0))
                    })
            else:
                # Multi ticker structure
                # yfinance returns MultiIndex columns if >1 ticker
                # columns: (Price, Ticker)
                
                # Check if columns are MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    # Iterate columns level 1 (Tickers)
                    # OR iterate valid cached list
                    # It's easier to iterate the batch and check if in data
                    for t in batch:
                        try:
                            df_t = data[t].dropna()
                            if df_t.empty: continue
                            
                            row = df_t.iloc[-1]
                            all_records.append({
                                'date': row.name,
                                'ticker': t,
                                'open': float(row.get('Open', 0)),
                                'high': float(row.get('High', 0)),
                                'low': float(row.get('Low', 0)),
                                'close': float(row.get('Close', 0)),
                                'volume': int(row.get('Volume', 0))
                            })
                        except KeyError:
                            pass # Ticker data missing
                else:
                    # Sometimes returns simple index if only 1 valid ticker found in list
                    pass 

        except Exception as e:
            logger.warning(f"  Batch {batch_num} failed: {e}")
            continue

    if not all_records:
        return pd.DataFrame()
        
    df_result = pd.DataFrame(all_records)
    logger.info(f"Fetched {len(df_result)} valid price records.")
    return df_result


def fetch_latest_news(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch latest news for tickers using existing news_loader.
    """
    try:
        from bronze.news_loader import load_market_news
        # Use 'auto' source (API if keys exist, else sample)
        # Fetching fewer articles for realtime update
        df = load_market_news(source='auto', num_articles=50, save_text_files=True)
        
        # Filter for relevant tickers if possible? 
        # news_loader fetches general news or ticker-specific if we change it.
        # Currently load_market_news fetches 'general' category from Finnhub.
        # We'll use result as is.
        return df
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        return pd.DataFrame()


def save_incremental(df: pd.DataFrame, prefix: str):
    """
    Save incremental DataFrame to Bronze Lakehouse with timestamp.
    Also saves raw JSON/TXT for manual inspection (User Request).
    """
    if df.empty:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parquet = f"{prefix}_{timestamp}.parquet"
    filename_json = f"{prefix}_{timestamp}.json"
    
    # Determine directory based on prefix
    if prefix == 'prices':
        lake_dir = BRONZE_DIR / 'prices_lakehouse'
        raw_dir = BRONZE_DIR / 'prices_raw_view' # For user visibility
    elif prefix == 'news':
        lake_dir = BRONZE_DIR / 'market_news_lakehouse'
        raw_dir = BRONZE_DIR / 'news_raw_view'   # For user visibility
    else:
        lake_dir = BRONZE_DIR / 'misc'
        raw_dir = BRONZE_DIR / 'misc_raw'

    lake_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save standard Parquet (Best practice for pipeline)
    out_path_parquet = lake_dir / filename_parquet
    df.to_parquet(out_path_parquet, index=False)
    
    # 2. Save JSON/TXT (For User Verification/Realtime Feel)
    out_path_raw = raw_dir / filename_json
    
    if prefix == 'prices':
        # Save pretty JSON for visibility
        df.head(50).to_json(out_path_raw, orient='records', date_format='iso', indent=2)
        logger.info(f"Saved RAW JSON (Top 50): {out_path_raw}")
        
    elif prefix == 'news':
        # Save as text files for diversity
        for idx, row in df.iterrows():
            txt_name = f"news_{timestamp}_{idx}.txt"
            txt_path = raw_dir / txt_name
            try:
                content = f"TITLE: {row.get('headline', 'No Title')}\n" \
                          f"DATE: {row.get('datetime', datetime.now())}\n" \
                          f"SOURCE: {row.get('source', 'Unknown')}\n" \
                          f"URL: {row.get('url', '')}\n" \
                          f"\nSUMMARY:\n{row.get('summary', '')}"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass
        logger.info(f"Saved RAW TXT News to: {raw_dir}")

    logger.info(f"Saved incremental {prefix}: {out_path_parquet} ({len(df)} rows)")
    return out_path_parquet


def run_realtime_loop(
    iterations: Optional[int] = REALTIME_MAX_ITERATIONS,
    price_interval: int = REALTIME_PRICE_INTERVAL,
    news_interval: int = REALTIME_NEWS_INTERVAL
):
    """
    Infinite loop to fetch data and sync to R2.
    """
    logger.info("Starting Real-time Sync Loop...")
    logger.info(f"Price Interval: {price_interval}s")
    logger.info(f"News Interval: {news_interval}s")
    
    # Load universe
    universe = get_universe()
    tickers = list(universe.get_intersection())
    
    if not tickers:
        # Fallback to loading from file if intersection is empty
        logger.warning("Universe intersection empty. Loading from Bronze prices.")
        from utils.ticker_universe import register_tickers_from_bronze
        register_tickers_from_bronze()
        tickers = list(universe.get_union())
        
    if not tickers:
        logger.error("No tickers found in universe. Exiting.")
        return

    logger.info(f"Tracking {len(tickers)} tickers: {tickers[:5]}...")

    last_price_fetch = 0
    last_news_fetch = 0
    
    count = 0
    
    while True:
        now = time.time()
        
        # --- PRICE UPDATE ---
        if now - last_price_fetch >= price_interval:
            logger.info("--- [PRICE UPDATE] ---")
            df_prices = fetch_latest_prices(tickers)
            if not df_prices.empty:
                save_incremental(df_prices, 'prices')
                # Upload Bronze (all changes) to R2
                # We upload the whole dir to be safe or optimize?
                # upload_bronze_to_r2 handles directory sync
                upload_bronze_to_r2(BRONZE_DIR)
            last_price_fetch = time.time()
            
        # --- NEWS UPDATE ---
        if now - last_news_fetch >= news_interval:
            logger.info("--- [NEWS UPDATE] ---")
            df_news = fetch_latest_news(tickers)
            if not df_news.empty:
                # News loader saves to lakehouse internally usually, 
                # but fetch_latest_news calls load_market_news which saves files?
                # Checking news_loader: load_market_news DOES save text files, 
                # but returns DF. We should save DF to parquet if needed.
                # Actually load_market_news does NOT save parquet lakehouse automatically inside the function?
                # Checked news_loader.py: load_market_news returns DF. 
                # CLI wrapper calls save_to_lakehouse.
                # So we must save here.
                from bronze.news_loader import save_to_lakehouse
                save_to_lakehouse(df_news)
                
                upload_bronze_to_r2(BRONZE_DIR)
            last_news_fetch = time.time()
            
        # Check iteration limit
        if iterations is not None:
            count += 1
            if count >= iterations:
                logger.info(f"Reached max iterations ({iterations}). Stopping.")
                break
        
        # Sleep small amount to prevent CPU spin
        time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=None, help="Number of loops")
    parser.add_argument("--price-interval", type=int, default=REALTIME_PRICE_INTERVAL)
    parser.add_argument("--news-interval", type=int, default=REALTIME_NEWS_INTERVAL)
    
    args = parser.parse_args()
    
    run_realtime_loop(
        iterations=args.iterations,
        price_interval=args.price_interval,
        news_interval=args.news_interval
    )
