"""
Silver Layer: Memory-optimized clean for free-tier VM (1GB RAM)

Processes data in batches by ticker to stay within memory limits.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from config import (
    BRONZE_DIR, SILVER_DIR, METADATA_DIR,
    PRICE_DATA_SCHEMA, SILVER_QUALITY_CHECKS,
    GICS_SECTORS, LOG_FORMAT
)

# Logging setup
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Constants
BRONZE_FILE = BRONZE_DIR / 'prices.parquet'
BRONZE_FILE_ALT = BRONZE_DIR / 'all_stock_data.parquet'
OUTPUT_PATH = SILVER_DIR / 'enriched_stocks.parquet'

# Memory optimization: process tickers in batches
BATCH_SIZE = 50  # Process 50 tickers at a time


def get_all_tickers(bronze_path):
    """Get list of all unique tickers without loading full dataset."""
    logger.info("Reading ticker list from file...")
    
    # Read only ticker column to get unique list
    df_tickers = pd.read_parquet(bronze_path, columns=['ticker'])
    unique_tickers = sorted(df_tickers['ticker'].unique())
    
    logger.info(f"Found {len(unique_tickers)} unique tickers")
    return unique_tickers


def process_ticker_batch(bronze_path, tickers, essential_cols):
    """Process a batch of tickers and return cleaned data."""
    # Read only selected tickers
    df = pd.read_parquet(bronze_path, columns=essential_cols)
    df.columns = [c.lower() for c in df.columns]
    
    # Filter to selected tickers
    df = df[df['ticker'].isin(tickers)]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Data cleaning steps
    # 1. Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Deduplicate
    df = df.sort_values('date').drop_duplicates(subset=['ticker', 'date'], keep='last')
    
    # 3. Remove nulls
    df = df.dropna(subset=['date', 'ticker', 'close', 'open', 'high', 'low', 'volume'])
    
    # 4. Quality gates
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['volume'] >= 0) & (df['high'] >= df['low'])]
    
    # 5. Calculate daily returns
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() *100
    df['daily_return'] = df['daily_return'].fillna(0)
    
    # 6. Add enrichment metadata
    df['enriched_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    
    # 7. Add quality flags
    df['is_penny_stock'] = df['close'] < 1.0
    df['has_outlier_return'] = df['daily_return'].abs() > 50
    df['low_liquidity'] = df['volume'] < 10000
    
    return df


def clean_silver_data_optimized():
    """Main Silver cleaning pipeline - memory optimized for 1GB RAM."""
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("SILVER LAYER PROCESSING STARTED (MEMORY-OPTIMIZED)")
    logger.info("=" * 70)
    
    try:
        # Find bronze file
        if BRONZE_FILE.exists():
            bronze_path = BRONZE_FILE
        elif BRONZE_FILE_ALT.exists():
            bronze_path = BRONZE_FILE_ALT
        else:
            raise FileNotFoundError(f"No bronze file found")
        
        logger.info(f"Processing: {bronze_path}")
        
        # Get file info
        parquet_file = pq.ParquetFile(bronze_path)
        total_rows = parquet_file.metadata.num_rows
        file_columns = parquet_file.schema.names
        logger.info(f"Total rows: {total_rows:,}")
        
        # Define essential columns
        essential_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        actual_cols = [col for col in file_columns if col.lower() in [c.lower() for c in essential_cols]]
        
        # Get all tickers
        all_tickers = get_all_tickers(bronze_path)
        
        # Process in batches
        all_batches = []
        total_tickers = len(all_tickers)
        num_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing {total_tickers} tickers in {num_batches} batches of {BATCH_SIZE}")
        
        for i in range(0, total_tickers, BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            ticker_batch = all_tickers[i:i+BATCH_SIZE]
            
            logger.info(f"[Batch {batch_num}/{num_batches}] Processing {len(ticker_batch)} tickers...")
            
            df_batch = process_ticker_batch(bronze_path, ticker_batch, actual_cols)
            
            if len(df_batch) > 0:
                all_batches.append(df_batch)
                logger.info(f"  → {len(df_batch):,} rows processed")
            
            # Free memory
            del df_batch
        
        # Combine all batches
        logger.info("Combining all batches...")
        df_final = pd.concat(all_batches, ignore_index=True)
        
        # Add sector info
        try:
            metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
            if metadata_file.exists():
                logger.info("Adding sector metadata...")
                metadata = pd.read_parquet(metadata_file)
                if 'ticker' in metadata.columns and 'sector' in metadata.columns:
                    merge_cols = ['ticker', 'sector']
                    if 'industry' in metadata.columns:
                        merge_cols.append('industry')
                    
                    df_final = df_final.merge(metadata[merge_cols], on='ticker', how='left')
                    df_final['sector'] = df_final['sector'].fillna('Unknown')
                    if 'industry' not in df_final.columns:
                        df_final['industry'] = 'Unknown'
                    else:
                        df_final['industry'] = df_final['industry'].fillna('Unknown')
            else:
                df_final['sector'] = 'Unknown'
                df_final['industry'] = 'Unknown'
        except Exception as e:
            logger.warning(f"Could not add sector metadata: {e}")
            df_final['sector'] = 'Unknown'
            df_final['industry'] = 'Unknown'
        
        # Add ticker-level quality flags
        ticker_counts = df_final.groupby('ticker').size()
        insufficient_tickers = ticker_counts[ticker_counts < 252].index
        df_final['insufficient_history'] = df_final['ticker'].isin(insufficient_tickers)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info("SILVER LAYER PROCESSING COMPLETED")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total rows: {len(df_final):,}")
        logger.info(f"Unique tickers: {df_final['ticker'].nunique():,}")
        logger.info(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        logger.info(f"Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info("=" * 70)
        
        return df_final
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"SILVER LAYER PROCESSING FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


def save_to_silver(df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
    """Save DataFrame to Silver layer."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {output_path}...")
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size = output_path.stat().st_size / 1024**2
        logger.info(f"[OK] Data saved to {output_path}")
        logger.info(f"[OK] File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise


def main():
    """CLI entry point."""
    logger.info("")
    logger.info("SILVER LAYER PROCESSING (MEMORY-OPTIMIZED FOR FREE-TIER VM)")
    logger.info("")
    
    try:
        # Process data
        df = clean_silver_data_optimized()
        
        # Save to Silver layer
        save_to_silver(df)
        
        logger.info("")
        logger.info("[OK] Silver layer processing completed successfully!")
        logger.info(f"[OK] Output: {OUTPUT_PATH}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"[ERR] Silver layer processing failed: {str(e)}")
        logger.error("")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
