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
def load_bronze_data(
    columns: list = None, 
    sample_tickers: int = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Load data from Bronze layer with memory optimization and partition support.
    
    Args:
        columns: Optional list of columns to load (reduces memory)
        sample_tickers: If set, only load this many random tickers (for testing)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    """
    # Check for partitioned Bronze first
    partitioned_dir = BRONZE_DIR / 'prices_partitioned'
    
    if partitioned_dir.exists():
        logger.info(f"Loading from partitioned Bronze: {partitioned_dir}")
        return load_partitioned_bronze(partitioned_dir, columns, start_date, end_date, sample_tickers)
    
    # Fallback to old single-file format
    if BRONZE_FILE.exists():
        bronze_path = BRONZE_FILE
    elif BRONZE_FILE_ALT.exists():
        bronze_path = BRONZE_FILE_ALT
    else:
        raise FileNotFoundError(
            f"No bronze file found. Expected: {BRONZE_FILE} or {BRONZE_FILE_ALT}"
        )
    
    logger.info(f"Loading data from {bronze_path}")
    
    # Read metadata first to get info
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(bronze_path)
    total_rows = parquet_file.metadata.num_rows
    file_columns = parquet_file.schema.names
    logger.info(f"[INFO] Total rows in file: {total_rows:,}")
    logger.info(f"[INFO] Columns in file: {file_columns}")
    
    # Define columns to load (only essential ones to save memory)
    # We normalized to lowercase in Bronze, but let's be flexible
    essential_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    
    # Map column names (case-insensitive)
    actual_cols = []
    for col in essential_cols:
        for file_col in file_columns:
            if file_col.lower() == col.lower():
                actual_cols.append(file_col)
                break
    
    if columns:
        # User specified columns - map them too
        for col in columns:
            for file_col in file_columns:
                if file_col.lower() == col.lower() and file_col not in actual_cols:
                    actual_cols.append(file_col)
                    break
    
    logger.info(f"[INFO] Loading columns: {actual_cols}")
    
    # Read with column selection
    df = pd.read_parquet(bronze_path, columns=actual_cols)
    
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Apply date filter if specified
    if start_date or end_date:
        df['date'] = pd.to_datetime(df['date'])
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        logger.info(f"[FILTER] Date range: {start_date} to {end_date}, rows: {len(df):,}")
    
    # Sample tickers if requested
    if sample_tickers:
        unique_tickers = df['ticker'].unique()
        if len(unique_tickers) > sample_tickers:
            sampled = pd.Series(unique_tickers).sample(sample_tickers, random_state=42).tolist()
            df = df[df['ticker'].isin(sampled)]
            logger.info(f"[SAMPLE] Sampled {sample_tickers} tickers, rows: {len(df):,}")
    
    logger.info(f"[OK] Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    logger.info(f"[OK] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


def load_partitioned_bronze(
    partition_dir: Path,
    columns: list = None,
    start_date: str = None,
    end_date: str = None,
    sample_tickers: int = None
) -> pd.DataFrame:
    """
    Load data from date-partitioned Bronze directory.
    
    Args:
        partition_dir: Path to partitioned directory
        columns: Columns to load
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        sample_tickers: Number of tickers to sample
    """
    # Get all partition directories
    partition_dirs = sorted([d for d in partition_dir.iterdir() if d.is_dir() and d.name.startswith('date=')])
    
    if not partition_dirs:
        raise FileNotFoundError(f"No partitions found in {partition_dir}")
    
    logger.info(f"Found {len(partition_dirs)} date partitions")
    
    # Filter partitions by date if specified
    if start_date or end_date:
        filtered_dirs = []
        for d in partition_dirs:
            date_str = d.name.replace('date=', '')
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            filtered_dirs.append(d)
        partition_dirs = filtered_dirs
        logger.info(f"Filtered to {len(partition_dirs)} partitions ({start_date} to {end_date})")
    
    # Load partitions in batches
    dfs = []
    for i, partition_dir_path in enumerate(partition_dirs):
        parquet_files = list(partition_dir_path.glob('*.parquet'))
        if not parquet_files:
            continue
        
        # Read partition
        df_partition = pd.read_parquet(parquet_files[0], columns=columns)
        dfs.append(df_partition)
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Loaded {i + 1}/{len(partition_dirs)} partitions")
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all partitions
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Sample tickers if requested
    if sample_tickers:
        unique_tickers = df['ticker'].unique()
        if len(unique_tickers) > sample_tickers:
            sampled = pd.Series(unique_tickers).sample(sample_tickers, random_state=42).tolist()
            df = df[df['ticker'].isin(sampled)]
            logger.info(f"[SAMPLE] Sampled {sample_tickers} tickers")
    
    logger.info(f"[OK] Loaded {len(df):,} rows from partitioned Bronze")
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
    """Add sector from metadata if available.
    
    Looks for sector data in:
    1. METADATA_DIR / 'ticker_metadata.parquet' (preferred)
    2. BRONZE_DIR / 'stock_metadata_lakehouse' (fallback)
    """
    from utils import is_lakehouse_table, lakehouse_to_pandas
    
    metadata = None
    
    # Option 1: Check ticker_metadata.parquet
    metadata_file = METADATA_DIR / 'ticker_metadata.parquet'
    if metadata_file.exists():
        logger.info("Loading sector metadata from ticker_metadata.parquet...")
        metadata = pd.read_parquet(metadata_file)
    
    # Option 2: Fallback to stock_metadata_lakehouse
    if metadata is None or len(metadata) == 0:
        lakehouse_path = BRONZE_DIR / 'stock_metadata_lakehouse'
        if is_lakehouse_table(lakehouse_path):
            logger.info("Loading sector metadata from stock_metadata_lakehouse...")
            metadata = lakehouse_to_pandas(lakehouse_path)
    
    # Apply metadata if available
    if metadata is not None and len(metadata) > 0:
        if 'ticker' in metadata.columns and 'sector' in metadata.columns:
            # Ensure we have the required columns
            merge_cols = ['ticker', 'sector']
            if 'industry' in metadata.columns:
                merge_cols.append('industry')
            
            df = df.merge(
                metadata[merge_cols],
                on='ticker',
                how='left'
            )
            df['sector'] = df['sector'].fillna('Unknown')
            if 'industry' not in df.columns:
                df['industry'] = 'Unknown'
            else:
                df['industry'] = df['industry'].fillna('Unknown')
            
            known_sectors = df[df['sector'] != 'Unknown']['sector'].nunique()
            logger.info(f"[OK] Added sector info for {known_sectors} unique sectors")
            logger.info(f"    Tickers with sector: {(df['sector'] != 'Unknown').sum():,}")
            logger.info(f"    Tickers without sector: {(df['sector'] == 'Unknown').sum():,}")
        else:
            logger.warning("[WARN] Metadata missing 'ticker' or 'sector' columns")
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
def main(incremental: bool = False, start_date: str = None):
    """
    CLI entry point.
    
    Args:
        incremental: If True, only process data newer than last run
        start_date: If set, only process data from this date onwards (YYYY-MM-DD)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Silver Layer Data Processing')
    parser.add_argument('--incremental', '-i', action='store_true',
                        help='Only process new data since last run')
    parser.add_argument('--start-date', '-s', type=str, default=None,
                        help='Process data from this date onwards (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Use args if called from CLI, otherwise use function params
    incremental = args.incremental or incremental
    start_date = args.start_date or start_date
    
    logger.info("")
    logger.info(" SILVER LAYER PROCESSING")
    if incremental:
        logger.info(" Mode: INCREMENTAL")
    if start_date:
        logger.info(f" Start Date: {start_date}")
    logger.info("")
    
    try:
        # Process data (filter by date if specified)
        df = clean_silver_data()
        
        # Apply date filter if incremental or start_date specified
        if start_date and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            original_len = len(df)
            df = df[df['date'] >= pd.to_datetime(start_date)]
            logger.info(f"[FILTER] Kept {len(df):,} of {original_len:,} rows (from {start_date})")
        
        if incremental:
            # Load existing Silver data and merge
            existing_path = OUTPUT_PATH
            if existing_path.exists():
                logger.info(f"[INCREMENTAL] Loading existing Silver data...")
                df_existing = pd.read_parquet(existing_path)
                # Only keep new dates
                if 'date' in df_existing.columns:
                    max_existing_date = pd.to_datetime(df_existing['date']).max()
                    df = df[pd.to_datetime(df['date']) > max_existing_date]
                    logger.info(f"[INCREMENTAL] Processing {len(df):,} new rows after {max_existing_date.date()}")
                    # Concat with existing
                    df = pd.concat([df_existing, df], ignore_index=True)
        
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

