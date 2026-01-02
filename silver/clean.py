"""
Silver Layer: Clean and enrich stock data

Handles deduplication, quality gates, daily returns calculation,
and joining with sector metadata.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from config import (
    BRONZE_DIR, SILVER_DIR, METADATA_DIR,
    PRICE_DATA_SCHEMA, SILVER_QUALITY_CHECKS,
    GICS_SECTORS, LOG_FORMAT
)

# =============================================================================
# LOGGING SETUP 
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
BRONZE_FILE = BRONZE_DIR / 'prices.parquet'
BRONZE_FILE_ALT = BRONZE_DIR / 'all_stock_data.parquet'
OUTPUT_PATH = SILVER_DIR / 'enriched_stocks.parquet'

# Column name mapping (Kaggle PascalCase → standard lowercase)
COLUMN_MAPPING = {
    'Date': 'date',
    'Ticker': 'ticker', 
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Adj Close': 'adj_close',
    'ingested_at': 'ingested_at'
}


# =============================================================================
# LOAD DATA
# =============================================================================
def load_bronze_data(columns: list = None, sample_tickers: int = None) -> pd.DataFrame:
    """Load data from Bronze layer with memory optimization.
    
    Args:
        columns: Optional list of columns to load (reduces memory)
        sample_tickers: If set, only load this many random tickers (for testing)
    """
    if BRONZE_FILE.exists():
        bronze_path = BRONZE_FILE
    elif BRONZE_FILE_ALT.exists():
        bronze_path = BRONZE_FILE_ALT
    else:
        raise FileNotFoundError(
            f"No bronze file found. Expected: {BRONZE_FILE} or {BRONZE_FILE_ALT}"
        )
    
    logger.info(f"Loading data from {bronze_path}")
    
    # Use PyArrow for memory-efficient reading
    import pyarrow.parquet as pq
    
    # Read metadata first to get info
    parquet_file = pq.ParquetFile(bronze_path)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"[INFO] Total rows in file: {total_rows:,}")
    
    # Define columns to load (only essential ones to save memory)
    if columns is None:
        columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Read with PyArrow (more memory efficient than pandas directly)
    logger.info(f"[INFO] Loading columns: {columns}")
    table = parquet_file.read(columns=columns)
    df = table.to_pandas()
    
    # Sample tickers if requested (for testing)
    if sample_tickers:
        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'ticker'
        unique_tickers = df[ticker_col].unique()
        if len(unique_tickers) > sample_tickers:
            import random
            sampled = random.sample(list(unique_tickers), sample_tickers)
            df = df[df[ticker_col].isin(sampled)]
            logger.info(f"[INFO] Sampled {sample_tickers} tickers for testing")
    
    logger.info(f"[OK] Loaded {len(df):,} rows from Bronze layer")
    logger.info(f"[OK] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase."""
    # Create mapping for current columns
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]
        elif col.lower() != col:
            rename_map[col] = col.lower()
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"[OK] Standardized column names: {list(rename_map.keys())} → {list(rename_map.values())}")
    else:
        logger.info("[OK] Column names already standardized")
    
    return df


# DATA CLEANING
# =============================================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (ticker, date) pairs."""
    initial_rows = len(df)
    
    df = df.sort_values('date').drop_duplicates(
        subset=['ticker', 'date'],
        keep='last'
    )
    
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"[OK] Removed {removed:,} duplicate rows")
    else:
        logger.info("[OK] No duplicates found")
    
    return df


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column is datetime type."""
    if df['date'].dtype == 'object':
        logger.info("Converting date column to datetime...")
        df['date'] = pd.to_datetime(df['date'])
        logger.info("[OK] Date column converted to datetime")
    
    return df


def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with nulls in critical columns."""
    initial_rows = len(df)
    
    critical_cols = ['date', 'ticker', 'close', 'open', 'high', 'low', 'volume']
    existing_cols = [c for c in critical_cols if c in df.columns]
    
    # Count nulls before
    null_counts = df[existing_cols].isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        logger.info(f"Found {total_nulls:,} null values in critical columns")
        for col in existing_cols:
            if null_counts[col] > 0:
                logger.info(f"  - {col}: {null_counts[col]:,} nulls")
        
        # Remove rows with any nulls in critical columns
        df = df.dropna(subset=existing_cols)
        removed = initial_rows - len(df)
        logger.info(f"[OK] Removed {removed:,} rows with null values")
    else:
        logger.info("[OK] No null values in critical columns")
    
    return df


def apply_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Filter invalid rows (close<=0, high<low, etc)."""
    initial_rows = len(df)
    
    # Rule 1: close > 0 (critical)
    invalid_close = df[df['close'] <= 0]
    if len(invalid_close) > 0:
        logger.warning(f"[WARN]  Removing {len(invalid_close):,} rows with close <= 0")
        df = df[df['close'] > 0]
    
    # Rule 2: high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        logger.warning(f"[WARN]  Removing {len(invalid_hl):,} rows with high < low")
        df = df[df['high'] >= df['low']]
    
    # Rule 3: volume >= 0
    invalid_vol = df[df['volume'] < 0]
    if len(invalid_vol) > 0:
        logger.warning(f"[WARN]  Removing {len(invalid_vol):,} rows with volume < 0")
        df = df[df['volume'] >= 0]
    
    # Rule 4: open > 0
    invalid_open = df[df['open'] <= 0]
    if len(invalid_open) > 0:
        logger.warning(f"[WARN]  Removing {len(invalid_open):,} rows with open <= 0")
        df = df[df['open'] > 0]
    
    removed = initial_rows - len(df)
    logger.info(f"[OK] Quality gates: removed {removed:,} invalid rows total")
    logger.info(f"[OK] Remaining rows: {len(df):,}")
    
    return df


def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily pct_change for each ticker."""
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate daily return (percentage)
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() * 100
    
    # Fill NaN for first day of each ticker with 0
    df['daily_return'] = df['daily_return'].fillna(0)
    
    logger.info(f"[OK] Calculated daily returns for {df['ticker'].nunique():,} tickers")
    logger.info(f"  Mean return: {df['daily_return'].mean():.4f}%")
    logger.info(f"  Std return: {df['daily_return'].std():.4f}%")
    
    return df


def add_sector_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector from metadata if available."""
    metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
    
    if metadata_file.exists():
        logger.info("Loading sector metadata...")
        metadata = pd.read_parquet(metadata_file)
        
        if 'ticker' in metadata.columns and 'sector' in metadata.columns:
            df = df.merge(
                metadata[['ticker', 'sector', 'industry']],
                on='ticker',
                how='left'
            )
            df['sector'] = df['sector'].fillna('Unknown')
            df['industry'] = df['industry'].fillna('Unknown')
            logger.info(f"[OK] Added sector info for {df['sector'].nunique()} unique sectors")
        else:
            logger.warning("[WARN]  Metadata file missing 'ticker' or 'sector' columns")
            df['sector'] = 'Unknown'
            df['industry'] = 'Unknown'
    else:
        logger.info("No metadata file found, sector will be 'Unknown'")
        df['sector'] = 'Unknown'
        df['industry'] = 'Unknown'
    
    return df


def add_enrichment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add enriched_at timestamp."""
    df['enriched_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    
    logger.info(f"[OK] Added enrichment metadata")
    
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for data quality issues."""
    logger.info("Adding quality flags (preserving all data)...")
    
    # Flag 1: Penny stocks (close < $1)
    df['is_penny_stock'] = df['close'] < 1.0
    penny_count = df['is_penny_stock'].sum()
    logger.info(f"  is_penny_stock: {penny_count:,} rows ({penny_count/len(df)*100:.2f}%)")
    
    # Flag 2: Outlier returns (|return| > 50%)
    # These are often stock splits, reverse splits, or data errors
    df['has_outlier_return'] = df['daily_return'].abs() > 50
    outlier_count = df['has_outlier_return'].sum()
    logger.info(f"  has_outlier_return: {outlier_count:,} rows ({outlier_count/len(df)*100:.2f}%)")
    
    # Flag 3: Low liquidity (volume < 10,000)
    df['low_liquidity'] = df['volume'] < 10000
    low_liq_count = df['low_liquidity'].sum()
    logger.info(f"  low_liquidity: {low_liq_count:,} rows ({low_liq_count/len(df)*100:.2f}%)")
    
    # Flag 4: Insufficient history per ticker (< 252 trading days)
    ticker_counts = df.groupby('ticker').size()
    insufficient_tickers = ticker_counts[ticker_counts < 252].index
    df['insufficient_history'] = df['ticker'].isin(insufficient_tickers)
    insuff_count = df['insufficient_history'].sum()
    logger.info(f"  insufficient_history: {insuff_count:,} rows ({len(insufficient_tickers)} tickers < 1 year)")
    
    # Summary: Count of "high quality" rows (no flags)
    high_quality_mask = (
        ~df['is_penny_stock'] & 
        ~df['has_outlier_return'] & 
        ~df['low_liquidity'] & 
        ~df['insufficient_history']
    )
    high_quality_count = high_quality_mask.sum()
    logger.info(f"  [OK] High-quality rows (no flags): {high_quality_count:,} ({high_quality_count/len(df)*100:.1f}%)")
    
    return df


# =============================================================================
# MAIN CLEANING FUNCTION
# =============================================================================
def clean_silver_data() -> pd.DataFrame:
    """Main Silver cleaning pipeline."""
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("SILVER LAYER PROCESSING STARTED")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load Bronze data
        df = load_bronze_data()
        
        # Step 2: Standardize columns
        df = standardize_columns(df)
        
        # Step 3: Convert date column to datetime
        df = convert_date_column(df)
        
        # Step 4: Deduplicate
        df = deduplicate(df)
        
        # Step 5: Remove null rows
        df = remove_null_rows(df)
        
        # Step 6: Quality gates
        df = apply_quality_gates(df)
        
        # Step 7: Calculate daily returns
        df = calculate_daily_return(df)
        
        # Step 8: Add sector info
        df = add_sector_info(df)
        
        # Step 9: Add metadata
        df = add_enrichment_metadata(df)
        
        # Step 10: Add quality flags (instead of deleting data)
        # Adds columns: is_penny_stock, has_outlier_return, low_liquidity, insufficient_history
        df = add_quality_flags(df)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info(" SILVER LAYER PROCESSING COMPLETED ")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Unique tickers: {df['ticker'].nunique():,}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("=" * 70)
        
        return df
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"[ERR] SILVER LAYER PROCESSING FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


def save_to_silver(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save DataFrame to Silver layer."""
    try:
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
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


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """CLI entry point."""
    logger.info("")
    logger.info(" SILVER LAYER PROCESSING")
    logger.info("")
    
    try:
        # Process data
        df = clean_silver_data()
        
        # Save to Silver layer
        save_to_silver(df)
        
        logger.info("")
        logger.info("[OK] Silver layer processing completed successfully!")
        logger.info(f"[OK] Output: {OUTPUT_PATH}")
        logger.info("")
        logger.info("Next step: Run Gold Layer (python gold/sector_analysis.py)")
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
