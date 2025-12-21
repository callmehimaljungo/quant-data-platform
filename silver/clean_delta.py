"""
Silver Layer: Data Lakehouse Version
Migrate from Parquet to Lakehouse format with all Silver transformations

This script:
1. Reads from Bronze Lakehouse (or Parquet fallback)
2. Applies Silver transformations (clean, dedupe, quality gates)
3. Saves to Silver Lakehouse format

Features enabled by Lakehouse:
- ACID transactions
- Time Travel (versioning)
- Schema tracking
- Incremental updates

Usage:
    python silver/clean_delta.py           # Convert existing Silver to Lakehouse
    python silver/clean_delta.py process   # Process from Bronze → Silver (Lakehouse)
"""

import sys
from pathlib import Path

# Add project root to path
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
    LOG_FORMAT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
# Bronze paths
BRONZE_LAKEHOUSE_PATH = BRONZE_DIR / 'prices_lakehouse'
BRONZE_PARQUET_PATH = BRONZE_DIR / 'all_stock_data.parquet'

# Silver paths
SILVER_PARQUET_PATH = SILVER_DIR / 'enriched_stocks.parquet'
SILVER_LAKEHOUSE_PATH = SILVER_DIR / 'enriched_lakehouse'

# Column mapping
COLUMN_MAPPING = {
    'Date': 'date',
    'Ticker': 'ticker', 
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Dividends': 'dividends',
    'Stock Splits': 'stock_splits',
}


# =============================================================================
# LOAD DATA
# =============================================================================
def load_from_bronze() -> pd.DataFrame:
    """Load data from Bronze layer (Lakehouse or Parquet)"""
    from utils import is_lakehouse_table, lakehouse_to_pandas
    
    if is_lakehouse_table(BRONZE_LAKEHOUSE_PATH):
        logger.info(f"Loading from Bronze Lakehouse: {BRONZE_LAKEHOUSE_PATH}")
        return lakehouse_to_pandas(BRONZE_LAKEHOUSE_PATH)
    elif BRONZE_PARQUET_PATH.exists():
        logger.info(f"Bronze Lakehouse not found, falling back to Parquet: {BRONZE_PARQUET_PATH}")
        return pd.read_parquet(BRONZE_PARQUET_PATH)
    else:
        raise FileNotFoundError("No Bronze data found (Lakehouse or Parquet)")


# =============================================================================
# TRANSFORMATIONS
# =============================================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase"""
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]
        elif col.lower() != col:
            rename_map[col] = col.lower()
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"✓ Standardized {len(rename_map)} column names")
    
    return df


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column is datetime"""
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'])
        logger.info("✓ Converted date column to datetime")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates"""
    initial = len(df)
    df = df.sort_values('date').drop_duplicates(subset=['ticker', 'date'], keep='last')
    removed = initial - len(df)
    if removed > 0:
        logger.info(f"✓ Removed {removed:,} duplicate rows")
    return df


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with nulls in critical columns"""
    initial = len(df)
    critical = ['date', 'ticker', 'close', 'open', 'high', 'low', 'volume']
    existing = [c for c in critical if c in df.columns]
    df = df.dropna(subset=existing)
    removed = initial - len(df)
    if removed > 0:
        logger.info(f"✓ Removed {removed:,} rows with null values")
    return df


def apply_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters"""
    initial = len(df)
    
    # close > 0
    df = df[df['close'] > 0]
    # high >= low
    df = df[df['high'] >= df['low']]
    # volume >= 0
    df = df[df['volume'] >= 0]
    # open > 0
    df = df[df['open'] > 0]
    
    removed = initial - len(df)
    logger.info(f"✓ Quality gates: removed {removed:,} invalid rows")
    logger.info(f"✓ Remaining: {len(df):,} rows")
    
    return df


def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns"""
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() * 100
    df['daily_return'] = df['daily_return'].fillna(0)
    logger.info(f"✓ Calculated daily returns for {df['ticker'].nunique():,} tickers")
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add enrichment metadata"""
    df['sector'] = 'Unknown'
    df['industry'] = 'Unknown'
    df['enriched_at'] = datetime.now()
    df['data_version'] = 'silver_lakehouse_v1'
    logger.info("✓ Added enrichment metadata")
    return df


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def process_silver_lakehouse() -> pd.DataFrame:
    """
    Full Silver processing pipeline with Lakehouse output
    """
    from utils import pandas_to_lakehouse, show_history
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("SILVER LAYER: LAKEHOUSE PROCESSING")
    logger.info("=" * 70)
    
    # Step 1: Load from Bronze
    df = load_from_bronze()
    logger.info(f"✓ Loaded {len(df):,} rows from Bronze")
    
    # Step 2: Standardize columns
    df = standardize_columns(df)
    
    # Step 3: Convert dates
    df = convert_date_column(df)
    
    # Step 4: Deduplicate
    df = deduplicate(df)
    
    # Step 5: Remove nulls
    df = remove_nulls(df)
    
    # Step 6: Quality gates
    df = apply_quality_gates(df)
    
    # Step 7: Daily returns
    df = calculate_daily_return(df)
    
    # Step 8: Add metadata
    df = add_metadata(df)
    
    # Step 9: Save to Lakehouse
    logger.info(f"\nSaving to Lakehouse: {SILVER_LAKEHOUSE_PATH}")
    pandas_to_lakehouse(df, SILVER_LAKEHOUSE_PATH, mode="overwrite")
    
    # Show history
    logger.info("\n--- Lakehouse History ---")
    show_history(SILVER_LAKEHOUSE_PATH, limit=5)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ SILVER LAKEHOUSE PROCESSING COMPLETED ✓✓✓")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Tickers: {df['ticker'].nunique():,}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Output: {SILVER_LAKEHOUSE_PATH}")
    logger.info("=" * 70)
    
    return df


def convert_silver_to_lakehouse():
    """
    Convert existing Silver Parquet to Lakehouse (no reprocessing)
    """
    from utils import pandas_to_lakehouse, show_history
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("SILVER LAYER: PARQUET -> LAKEHOUSE MIGRATION")
    logger.info("=" * 70)
    
    if not SILVER_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Silver Parquet not found: {SILVER_PARQUET_PATH}")
    
    # Load existing Silver
    logger.info(f"Loading from Parquet: {SILVER_PARQUET_PATH}")
    df = pd.read_parquet(SILVER_PARQUET_PATH)
    logger.info(f"✓ Loaded {len(df):,} rows")
    
    # Update metadata
    df['lakehouse_migrated_at'] = datetime.now()
    if 'data_version' in df.columns:
        df['data_version'] = 'silver_lakehouse_v1'
    
    # Save as Lakehouse
    logger.info(f"Converting to Lakehouse: {SILVER_LAKEHOUSE_PATH}")
    pandas_to_lakehouse(df, SILVER_LAKEHOUSE_PATH, mode="overwrite")
    
    # Show history
    show_history(SILVER_LAKEHOUSE_PATH, limit=5)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ SILVER LAKEHOUSE MIGRATION COMPLETED ✓✓✓")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Output: {SILVER_LAKEHOUSE_PATH}")
    logger.info("=" * 70)


def main():
    """Main execution"""
    mode = sys.argv[1] if len(sys.argv) > 1 else 'convert'
    
    logger.info("")
    logger.info("🚀 SILVER LAYER - DATA LAKEHOUSE")
    logger.info(f"📊 Mode: {mode}")
    logger.info("")
    
    try:
        if mode == 'process':
            # Full processing from Bronze
            process_silver_lakehouse()
        else:
            # Just convert existing Parquet
            convert_silver_to_lakehouse()
        
        logger.info("")
        logger.info("✅ Silver Lakehouse completed successfully!")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
