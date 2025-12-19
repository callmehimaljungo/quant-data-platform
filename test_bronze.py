"""
Test script for Bronze Layer
Validates ingestion results and data quality
"""

import pandas as pd
import logging
from pathlib import Path
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
    
    bronze_file = BRONZE_DIR / 'prices.parquet'
    
    # Check 1: File exists
    if not bronze_file.exists():
        logger.error(f"❌ FAILED: File not found - {bronze_file}")
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
    
    # Check 3: Schema validation
    logger.info("\n--- Schema Validation ---")
    
    required_columns = list(PRICE_DATA_SCHEMA.keys()) + ['ingested_at']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"❌ FAILED: Missing columns - {missing_cols}")
        return False
    logger.info(f"✓ All required columns present: {df.columns.tolist()}")
    
    # Check data types
    type_checks = []
    for col, expected_dtype in PRICE_DATA_SCHEMA.items():
        actual_dtype = str(df[col].dtype)
        match = False
        
        if expected_dtype == 'float64' and actual_dtype in ['float32', 'float64']:
            match = True
        elif expected_dtype == 'int64' and actual_dtype in ['int32', 'int64']:
            match = True
        elif expected_dtype == actual_dtype:
            match = True
        
        status = "✓" if match else "❌"
        type_checks.append(match)
        logger.info(f"  {status} {col}: {actual_dtype} (expected: {expected_dtype})")
    
    if not all(type_checks):
        logger.error("❌ FAILED: Data type mismatches detected")
        return False
    
    # Check 4: Data quality metrics
    logger.info("\n--- Data Quality Metrics ---")
    
    # Unique tickers
    n_tickers = df['ticker'].nunique()
    logger.info(f"✓ Unique tickers: {n_tickers:,}")
    
    # Date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    logger.info(f"✓ Date range: {min_date} to {max_date}")
    
    # Null counts
    logger.info("\n  Null counts by column:")
    null_counts = df.isnull().sum()
    for col in df.columns:
        null_pct = (null_counts[col] / len(df)) * 100
        status = "✓" if null_counts[col] == 0 else "⚠️"
        logger.info(f"    {status} {col}: {null_counts[col]:,} ({null_pct:.2f}%)")
    
    # Critical columns should have no nulls
    critical_cols = ['date', 'ticker', 'close']
    critical_nulls = {col: null_counts[col] for col in critical_cols if null_counts[col] > 0}
    if critical_nulls:
        logger.error(f"❌ FAILED: Nulls in critical columns - {critical_nulls}")
        return False
    logger.info(f"✓ No nulls in critical columns: {critical_cols}")
    
    # Sample data check
    logger.info("\n--- Sample Data (First 5 rows) ---")
    print(df.head())
    
    # Summary statistics
    logger.info("\n--- Summary Statistics ---")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
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
    
    bronze_file = BRONZE_DIR / 'prices.parquet'
    df = pd.read_parquet(bronze_file)
    
    # Check 1: High >= Low
    invalid_highs = df[df['high'] < df['low']]
    if len(invalid_highs) > 0:
        logger.warning(f"⚠️  {len(invalid_highs)} rows where high < low")
        logger.warning("  (Will be cleaned in Silver layer)")
    else:
        logger.info("✓ All rows: high >= low")
    
    # Check 2: Prices > 0
    zero_prices = df[(df['close'] <= 0) | (df['open'] <= 0)]
    if len(zero_prices) > 0:
        logger.warning(f"⚠️  {len(zero_prices)} rows with zero/negative prices")
        logger.warning("  (Will be cleaned in Silver layer)")
    else:
        logger.info("✓ All prices > 0")
    
    # Check 3: Volume >= 0
    negative_volume = df[df['volume'] < 0]
    if len(negative_volume) > 0:
        logger.error(f"❌ {len(negative_volume)} rows with negative volume")
    else:
        logger.info("✓ All volumes >= 0")
    
    # Check 4: Date continuity by ticker
    logger.info("\n✓ Date continuity check (sample 5 tickers):")
    sample_tickers = df['ticker'].unique()[:5]
    for ticker in sample_tickers:
        ticker_df = df[df['ticker'] == ticker].sort_values('date')
        date_gaps = ticker_df['date'].diff().dt.days.dropna()
        max_gap = date_gaps.max()
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
