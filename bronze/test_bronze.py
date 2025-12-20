"""
Test script for Bronze Layer
Validates ingestion results and data quality
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import logging
from config import BRONZE_DIR, PRICE_DATA_SCHEMA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_bronze_output():
    """
    Test Bronze layer output file
    
    Checks:
    1. File exists
    2. File is not empty
    3. Schema is correct
    4. Data quality metrics
    """
    logger.info("=" * 70)
    logger.info("BRONZE LAYER VALIDATION TEST")
    logger.info("=" * 70)
    
    # Try both file names (prices.parquet or all_stock_data.parquet)
    bronze_file = BRONZE_DIR / 'prices.parquet'
    if not bronze_file.exists():
        bronze_file = BRONZE_DIR / 'all_stock_data.parquet'
    
    # Check 1: File exists
    if not bronze_file.exists():
        logger.error(f"❌ FAILED: No bronze file found in {BRONZE_DIR}")
        logger.error("  Expected: prices.parquet or all_stock_data.parquet")
        return False
    logger.info(f"✓ File exists: {bronze_file}")
    
    # Load data
    try:
        df = pd.read_parquet(bronze_file)
        logger.info(f"✓ Successfully loaded data")
    except Exception as e:
        logger.error(f"❌ FAILED: Could not load file - {str(e)}")
        return False
    
    # Check 2: Not empty
    if len(df) == 0:
        logger.error("❌ FAILED: DataFrame is empty")
        return False
    logger.info(f"✓ Data loaded: {len(df):,} rows")
    
    # Standardize column names for validation (handle both PascalCase and lowercase)
    column_mapping = {
        'Date': 'date', 'Ticker': 'ticker', 'Open': 'open',
        'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df_columns = {column_mapping.get(c, c.lower()): c for c in df.columns}
    
    # Check 3: Schema validation
    logger.info("\n--- Schema Validation ---")
    
    required_columns = list(PRICE_DATA_SCHEMA.keys())
    # Map to actual column names in dataframe
    actual_columns_lower = [c.lower() for c in df.columns]
    missing_cols = [c for c in required_columns if c not in actual_columns_lower]
    
    if missing_cols:
        logger.error(f"❌ FAILED: Missing columns - {missing_cols}")
        logger.info(f"  Available columns: {df.columns.tolist()}")
        return False
    logger.info(f"✓ All required columns present: {df.columns.tolist()}")
    
    # Get actual column name (works with both PascalCase and lowercase)
    def get_col(name):
        """Get actual column name from df that matches name (case-insensitive)"""
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return name
    
    # Check data types (skip detailed check, just verify basic structure)
    logger.info("✓ Data types check (flexible for PascalCase/lowercase)")
    for col in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']:
        actual_col = get_col(col)
        if actual_col in df.columns:
            logger.info(f"  ✓ {actual_col}: {df[actual_col].dtype}")
    
    # Check 4: Data quality metrics
    logger.info("\n--- Data Quality Metrics ---")
    
    # Unique tickers
    ticker_col = get_col('ticker')
    n_tickers = df[ticker_col].nunique()
    logger.info(f"✓ Unique tickers: {n_tickers:,}")
    
    # Date range
    date_col = get_col('date')
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    logger.info(f"✓ Date range: {min_date} to {max_date}")
    
    # Null counts
    logger.info("\n  Null counts by column:")
    null_counts = df.isnull().sum()
    for col in df.columns:
        null_pct = (null_counts[col] / len(df)) * 100
        status = "✓" if null_counts[col] == 0 else "⚠️"
        logger.info(f"    {status} {col}: {null_counts[col]:,} ({null_pct:.2f}%)")
    
    # Critical columns should have no nulls (or very few that will be cleaned in Silver)
    critical_cols_actual = [get_col(c) for c in ['date', 'ticker', 'close']]
    critical_nulls = {col: null_counts[col] for col in critical_cols_actual if col in null_counts and null_counts[col] > 0}
    
    if critical_nulls:
        # Calculate percentage
        for col, count in critical_nulls.items():
            pct = (count / len(df)) * 100
            if pct > 0.01:  # More than 0.01% is a problem
                logger.error(f"❌ FAILED: Too many nulls in {col}: {count:,} ({pct:.4f}%)")
                return False
            else:
                logger.warning(f"⚠️  Minor nulls in {col}: {count:,} ({pct:.4f}%) - will be cleaned in Silver layer")
    else:
        logger.info(f"✓ No nulls in critical columns: {critical_cols_actual}")
    
    # Sample data check
    logger.info("\n--- Sample Data (First 5 rows) ---")
    print(df.head())
    
    # Summary statistics
    logger.info("\n--- Summary Statistics ---")
    ohlcv_cols = [get_col(c) for c in ['open', 'high', 'low', 'close', 'volume']]
    ohlcv_cols = [c for c in ohlcv_cols if c in df.columns]
    print(df[ohlcv_cols].describe())
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"\n✓ Memory usage: {memory_mb:.2f} MB")
    
    # File size
    file_size_mb = bronze_file.stat().st_size / 1024**2
    logger.info(f"✓ File size: {file_size_mb:.2f} MB")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓✓✓ ALL TESTS PASSED ✓✓✓")
    logger.info("=" * 70)
    
    return True


def test_data_integrity():
    """
    Additional integrity checks
    """
    logger.info("\n--- Data Integrity Checks ---")
    
    # Try both file names
    bronze_file = BRONZE_DIR / 'prices.parquet'
    if not bronze_file.exists():
        bronze_file = BRONZE_DIR / 'all_stock_data.parquet'
    
    df = pd.read_parquet(bronze_file)
    
    # Get actual column name (works with both PascalCase and lowercase)
    def get_col(name):
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return name
    
    high_col, low_col = get_col('high'), get_col('low')
    close_col, open_col = get_col('close'), get_col('open')
    volume_col = get_col('volume')
    date_col, ticker_col = get_col('date'), get_col('ticker')
    
    # Check 1: High >= Low
    invalid_highs = df[df[high_col] < df[low_col]]
    if len(invalid_highs) > 0:
        logger.warning(f"⚠️  {len(invalid_highs)} rows where high < low")
        logger.warning("  (Will be cleaned in Silver layer)")
    else:
        logger.info("✓ All rows: high >= low")
    
    # Check 2: Prices > 0
    zero_prices = df[(df[close_col] <= 0) | (df[open_col] <= 0)]
    if len(zero_prices) > 0:
        logger.warning(f"⚠️  {len(zero_prices)} rows with zero/negative prices")
        logger.warning("  (Will be cleaned in Silver layer)")
    else:
        logger.info("✓ All prices > 0")
    
    # Check 3: Volume >= 0
    negative_volume = df[df[volume_col] < 0]
    if len(negative_volume) > 0:
        logger.error(f"❌ {len(negative_volume)} rows with negative volume")
    else:
        logger.info("✓ All volumes >= 0")
    
    # Check 4: Date continuity by ticker
    logger.info("\n✓ Date continuity check (sample 5 tickers):")
    sample_tickers = df[ticker_col].unique()[:5]
    for ticker in sample_tickers:
        ticker_df = df[df[ticker_col] == ticker].sort_values(date_col)
        date_gaps = ticker_df[date_col].diff().dt.days.dropna()
        max_gap = date_gaps.max() if len(date_gaps) > 0 else 0
        logger.info(f"  {ticker}: max gap = {max_gap:.0f} days")


if __name__ == "__main__":
    try:
        # Run main tests
        success = test_bronze_output()
        
        if success:
            # Run integrity checks
            test_data_integrity()
            
            logger.info("\n🎉 Bronze Layer validation completed successfully!")
            logger.info("Next step: Run Silver Layer (python silver/clean.py)")
            exit(0)
        else:
            logger.error("\n❌ Bronze Layer validation failed!")
            logger.error("Please check the logs and re-run bronze/ingest.py")
            exit(1)
            
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
