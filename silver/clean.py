"""
Silver Layer: Clean and enrich stock data
Author: Quant Data Platform Team
Date: 2024

Purpose (Section 2.2 - Silver Layer):
- Deduplicate records (keep latest for each ticker/date)
- Quality gates: close > 0, high >= low, volume >= 0
- Standardize column names (PascalCase → lowercase)
- Calculate daily returns
- Join with sector metadata (if available)
- Output: enriched_stocks.parquet
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
# LOGGING SETUP (Section 7.2)
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
def load_bronze_data() -> pd.DataFrame:
    """
    Load data from Bronze layer
    
    Handles both file names (prices.parquet or all_stock_data.parquet)
    
    Returns:
        pd.DataFrame: Raw data from Bronze layer
        
    Raises:
        FileNotFoundError: If no bronze file found
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
    df = pd.read_parquet(bronze_path)
    logger.info(f"✓ Loaded {len(df):,} rows from Bronze layer")
    
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase
    
    Handles Kaggle's PascalCase format
    
    Args:
        df: DataFrame with potentially mixed case columns
        
    Returns:
        DataFrame with standardized lowercase column names
    """
    # Create mapping for current columns
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]
        elif col.lower() != col:
            rename_map[col] = col.lower()
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"✓ Standardized column names: {list(rename_map.keys())} → {list(rename_map.values())}")
    else:
        logger.info("✓ Column names already standardized")
    
    return df


# =============================================================================
# DATA CLEANING (Section 2.2 - Silver Transformations)
# =============================================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records
    
    Keeps last record for each (ticker, date) pair
    
    Args:
        df: DataFrame with potential duplicates
        
    Returns:
        DataFrame without duplicates
    """
    initial_rows = len(df)
    
    df = df.sort_values('date').drop_duplicates(
        subset=['ticker', 'date'],
        keep='last'
    )
    
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"✓ Removed {removed:,} duplicate rows")
    else:
        logger.info("✓ No duplicates found")
    
    return df


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date column is datetime type
    
    Args:
        df: DataFrame with 'date' column
        
    Returns:
        DataFrame with datetime date column
    """
    if df['date'].dtype == 'object':
        logger.info("Converting date column to datetime...")
        df['date'] = pd.to_datetime(df['date'])
        logger.info("✓ Date column converted to datetime")
    
    return df


def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with null values in critical columns
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame without null rows in critical columns
    """
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
        logger.info(f"✓ Removed {removed:,} rows with null values")
    else:
        logger.info("✓ No null values in critical columns")
    
    return df


def apply_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality gates (Section 2.2)
    
    Rules:
    - close > 0 (required)
    - high >= low
    - volume >= 0
    - open > 0
    
    Args:
        df: DataFrame to filter
        
    Returns:
        DataFrame with invalid rows removed
    """
    initial_rows = len(df)
    
    # Rule 1: close > 0 (critical)
    invalid_close = df[df['close'] <= 0]
    if len(invalid_close) > 0:
        logger.warning(f"⚠️  Removing {len(invalid_close):,} rows with close <= 0")
        df = df[df['close'] > 0]
    
    # Rule 2: high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        logger.warning(f"⚠️  Removing {len(invalid_hl):,} rows with high < low")
        df = df[df['high'] >= df['low']]
    
    # Rule 3: volume >= 0
    invalid_vol = df[df['volume'] < 0]
    if len(invalid_vol) > 0:
        logger.warning(f"⚠️  Removing {len(invalid_vol):,} rows with volume < 0")
        df = df[df['volume'] >= 0]
    
    # Rule 4: open > 0
    invalid_open = df[df['open'] <= 0]
    if len(invalid_open) > 0:
        logger.warning(f"⚠️  Removing {len(invalid_open):,} rows with open <= 0")
        df = df[df['open'] > 0]
    
    removed = initial_rows - len(df)
    logger.info(f"✓ Quality gates: removed {removed:,} invalid rows total")
    logger.info(f"✓ Remaining rows: {len(df):,}")
    
    return df


def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily return for each ticker
    
    daily_return = (close - prev_close) / prev_close * 100
    
    Args:
        df: DataFrame with 'ticker', 'date', 'close'
        
    Returns:
        DataFrame with 'daily_return' column added
    """
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate daily return (percentage)
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() * 100
    
    # Fill NaN for first day of each ticker with 0
    df['daily_return'] = df['daily_return'].fillna(0)
    
    logger.info(f"✓ Calculated daily returns for {df['ticker'].nunique():,} tickers")
    logger.info(f"  Mean return: {df['daily_return'].mean():.4f}%")
    logger.info(f"  Std return: {df['daily_return'].std():.4f}%")
    
    return df


def add_sector_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector information from metadata (if available)
    
    Args:
        df: DataFrame with 'ticker' column
        
    Returns:
        DataFrame with 'sector' column (or 'Unknown' if no metadata)
    """
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
            logger.info(f"✓ Added sector info for {df['sector'].nunique()} unique sectors")
        else:
            logger.warning("⚠️  Metadata file missing 'ticker' or 'sector' columns")
            df['sector'] = 'Unknown'
            df['industry'] = 'Unknown'
    else:
        logger.info("ℹ️  No metadata file found, sector will be 'Unknown'")
        df['sector'] = 'Unknown'
        df['industry'] = 'Unknown'
    
    return df


def add_enrichment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enrichment metadata
    
    Args:
        df: DataFrame to enrich
        
    Returns:
        DataFrame with metadata columns
    """
    df['enriched_at'] = datetime.now()
    df['data_version'] = 'silver_v1'
    
    logger.info(f"✓ Added enrichment metadata")
    
    return df


# =============================================================================
# MAIN CLEANING FUNCTION
# =============================================================================
def clean_silver_data() -> pd.DataFrame:
    """
    Main Silver layer cleaning pipeline
    
    Steps:
    1. Load from Bronze
    2. Standardize column names
    3. Deduplicate
    4. Apply quality gates
    5. Calculate daily returns
    6. Add sector info
    7. Add metadata
    
    Returns:
        pd.DataFrame: Cleaned and enriched data
    """
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
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info("✓✓✓ SILVER LAYER PROCESSING COMPLETED ✓✓✓")
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
        logger.error(f"❌ SILVER LAYER PROCESSING FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


def save_to_silver(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """
    Save DataFrame to Silver layer
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
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
        logger.info(f"✓ Data saved to {output_path}")
        logger.info(f"✓ File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Main execution function for Silver layer
    
    Usage:
        python silver/clean.py
    """
    logger.info("")
    logger.info("🚀 SILVER LAYER PROCESSING")
    logger.info("")
    
    try:
        # Process data
        df = clean_silver_data()
        
        # Save to Silver layer
        save_to_silver(df)
        
        logger.info("")
        logger.info("✅ Silver layer processing completed successfully!")
        logger.info(f"✅ Output: {OUTPUT_PATH}")
        logger.info("")
        logger.info("Next step: Run Gold Layer (python gold/sector_analysis.py)")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Silver layer processing failed: {str(e)}")
        logger.error("")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
