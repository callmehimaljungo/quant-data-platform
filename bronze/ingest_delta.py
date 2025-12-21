"""
Bronze Layer: Data Lakehouse Version
Migrate from Parquet to Lakehouse format (DuckDB-based)

This script converts existing Bronze Parquet data to Lakehouse format,
providing:
- ACID transactions
- Time Travel (versioning)
- Schema tracking
- Metadata management

Usage:
    python bronze/ingest_delta.py          # Convert existing Parquet to Lakehouse
    python bronze/ingest_delta.py kaggle   # Fresh ingest from Kaggle → Lakehouse
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import logging
from datetime import datetime
import pandas as pd

from config import BRONZE_DIR, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
PARQUET_PATH = BRONZE_DIR / 'all_stock_data.parquet'
LAKEHOUSE_PATH = BRONZE_DIR / 'prices_lakehouse'


def convert_bronze_to_lakehouse():
    """
    Convert existing Bronze Parquet to Lakehouse format
    """
    from utils import pandas_to_lakehouse, show_history
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: PARQUET -> LAKEHOUSE MIGRATION")
    logger.info("=" * 70)
    
    # Check if Parquet exists
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Bronze Parquet not found: {PARQUET_PATH}")
    
    # Load Parquet
    logger.info(f"Loading from Parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"✓ Loaded {len(df):,} rows")
    
    # Add migration metadata
    df['lakehouse_migrated_at'] = datetime.now()
    df['data_version'] = 'bronze_lakehouse_v1'
    
    # Save as Lakehouse
    logger.info(f"Converting to Lakehouse: {LAKEHOUSE_PATH}")
    pandas_to_lakehouse(df, LAKEHOUSE_PATH, mode="overwrite")
    
    # Show history
    logger.info("\n--- Lakehouse History ---")
    show_history(LAKEHOUSE_PATH, limit=5)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ BRONZE LAKEHOUSE MIGRATION COMPLETED ✓✓✓")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Output: {LAKEHOUSE_PATH}")
    logger.info("=" * 70)
    
    return LAKEHOUSE_PATH


def ingest_from_kaggle_to_lakehouse():
    """
    Fresh ingest from Kaggle directly to Lakehouse
    """
    from utils import pandas_to_lakehouse
    from bronze.ingest import ingest_from_kaggle, validate_schema
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("BRONZE LAYER: KAGGLE -> LAKEHOUSE INGESTION")
    logger.info("=" * 70)
    
    # Ingest from Kaggle
    df = ingest_from_kaggle()
    
    # Add metadata
    df['ingested_at'] = datetime.now()
    df['data_version'] = 'bronze_lakehouse_v1'
    
    # Validate schema
    validate_schema(df)
    
    # Save as Lakehouse
    logger.info(f"Saving to Lakehouse: {LAKEHOUSE_PATH}")
    pandas_to_lakehouse(df, LAKEHOUSE_PATH, mode="overwrite")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ BRONZE LAKEHOUSE INGESTION COMPLETED ✓✓✓")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Output: {LAKEHOUSE_PATH}")
    logger.info("=" * 70)
    
    return LAKEHOUSE_PATH


def main():
    """Main execution"""
    source = sys.argv[1] if len(sys.argv) > 1 else 'convert'
    
    logger.info("")
    logger.info("🚀 BRONZE LAYER - DATA LAKEHOUSE")
    logger.info(f"📊 Mode: {source}")
    logger.info("")
    
    try:
        if source == 'kaggle':
            ingest_from_kaggle_to_lakehouse()
        else:
            convert_bronze_to_lakehouse()
        
        logger.info("")
        logger.info("✅ Bronze Lakehouse completed successfully!")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
