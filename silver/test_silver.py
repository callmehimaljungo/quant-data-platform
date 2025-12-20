"""
Test script for Silver Layer
Validates cleaning results and data quality
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import logging
from config import SILVER_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_silver_output():
    """
    Test Silver layer output file
    
    Checks:
    1. File exists
    2. File is not empty
    3. Columns are standardized (lowercase)
    4. Quality gates passed
    5. Daily returns calculated
    """
    logger.info("=" * 70)
    logger.info("SILVER LAYER VALIDATION TEST")
    logger.info("=" * 70)
    
    silver_file = SILVER_DIR / 'enriched_stocks.parquet'
    
    # Check 1: File exists
    if not silver_file.exists():
        logger.error(f"❌ FAILED: File not found - {silver_file}")
        return False
    logger.info(f"✓ File exists: {silver_file}")
    
    # Load data
    try:
        df = pd.read_parquet(silver_file)
        logger.info(f"✓ Successfully loaded data")
    except Exception as e:
        logger.error(f"❌ FAILED: Could not load file - {str(e)}")
        return False
    
    # Check 2: Not empty
    if len(df) == 0:
        logger.error("❌ FAILED: DataFrame is empty")
        return False
    logger.info(f"✓ Data loaded: {len(df):,} rows")
    
    # Check 3: All columns lowercase
    non_lowercase = [c for c in df.columns if c != c.lower()]
    if non_lowercase:
        logger.warning(f"⚠️  Some columns not lowercase: {non_lowercase}")
    else:
        logger.info("✓ All column names standardized (lowercase)")
    
    # Check 4: Required columns present
    required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'daily_return']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"❌ FAILED: Missing columns - {missing_cols}")
        return False
    logger.info(f"✓ All required columns present")
    
    # Check 5: Quality gates
    logger.info("\n--- Quality Gate Checks ---")
    
    # close > 0
    invalid_close = df[df['close'] <= 0]
    if len(invalid_close) > 0:
        logger.error(f"❌ FAILED: {len(invalid_close)} rows with close <= 0")
        return False
    logger.info("✓ All close prices > 0")
    
    # high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        logger.error(f"❌ FAILED: {len(invalid_hl)} rows with high < low")
        return False
    logger.info("✓ All rows: high >= low")
    
    # volume >= 0
    invalid_vol = df[df['volume'] < 0]
    if len(invalid_vol) > 0:
        logger.error(f"❌ FAILED: {len(invalid_vol)} rows with volume < 0")
        return False
    logger.info("✓ All volume >= 0")
    
    # Check 6: Daily return calculated
    logger.info("\n--- Daily Return Checks ---")
    logger.info(f"✓ Daily return range: {df['daily_return'].min():.2f}% to {df['daily_return'].max():.2f}%")
    logger.info(f"✓ Mean daily return: {df['daily_return'].mean():.4f}%")
    logger.info(f"✓ Std daily return: {df['daily_return'].std():.4f}%")
    
    # Check 7: No duplicates
    duplicates = df.duplicated(subset=['ticker', 'date']).sum()
    if duplicates > 0:
        logger.error(f"❌ FAILED: {duplicates} duplicate (ticker, date) pairs")
        return False
    logger.info("✓ No duplicate (ticker, date) pairs")
    
    # Summary statistics
    logger.info("\n--- Data Summary ---")
    logger.info(f"✓ Unique tickers: {df['ticker'].nunique():,}")
    logger.info(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"✓ Sectors: {df['sector'].nunique() if 'sector' in df.columns else 'N/A'}")
    
    # Sample data
    logger.info("\n--- Sample Data (First 5 rows) ---")
    print(df[['date', 'ticker', 'close', 'daily_return', 'sector']].head() if 'sector' in df.columns else df[['date', 'ticker', 'close', 'daily_return']].head())
    
    # File size
    file_size_mb = silver_file.stat().st_size / 1024**2
    logger.info(f"\n✓ File size: {file_size_mb:.2f} MB")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓✓✓ ALL SILVER LAYER TESTS PASSED ✓✓✓")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_silver_output()
        
        if success:
            logger.info("\n🎉 Silver Layer validation completed successfully!")
            logger.info("Next step: Run Gold Layer (python gold/sector_analysis.py)")
            exit(0)
        else:
            logger.error("\n❌ Silver Layer validation failed!")
            logger.error("Please check the logs and re-run silver/clean.py")
            exit(1)
            
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
