"""
Data Migration Script
Convert existing Bronze/Silver data to date-partitioned structure

Usage:
    python scripts/migrate_to_partitions.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BRONZE_DIR, SILVER_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_bronze_prices():
    """Convert Bronze prices to date-partitioned structure using chunked processing"""
    logger.info("=" * 70)
    logger.info("MIGRATING BRONZE PRICES TO PARTITIONED FORMAT")
    logger.info("=" * 70)
    
    # Find source data
    source_paths = [
        BRONZE_DIR / 'prices_lakehouse',
        BRONZE_DIR / 'prices.parquet',
        BRONZE_DIR / 'all_stock_data.parquet',
    ]
    
    source_path = None
    for path in source_paths:
        if path.exists():
            if path.is_dir():
                parquet_files = list(path.glob('*.parquet'))
                if parquet_files:
                    source_path = parquet_files[0]
                    break
            else:
                source_path = path
                break
    
    if source_path is None:
        logger.error("No Bronze price data found!")
        return False
    
    logger.info(f"Source: {source_path}")
    
    # Create output directory
    output_dir = BRONZE_DIR / 'prices_partitioned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use PyArrow for chunked reading
    parquet_file = pq.ParquetFile(source_path)
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total rows: {total_rows:,}")
    
    # Get column names (normalize to lowercase)
    schema = parquet_file.schema_arrow
    col_names = [name.lower() for name in schema.names]
    
    # Find date column
    date_col = None
    for name in schema.names:
        if name.lower() == 'date':
            date_col = name
            break
    
    if date_col is None:
        logger.error("No 'date' column found!")
        return False
    
    # Process in row group batches
    num_row_groups = parquet_file.num_row_groups
    logger.info(f"Processing {num_row_groups} row groups...")
    
    dates_written = set()
    
    for i in range(num_row_groups):
        # Read one row group at a time
        table = parquet_file.read_row_group(i)
        df = table.to_pandas()
        
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and write partitions
        for date, group in df.groupby(df['date'].dt.date):
            date_str = date.strftime('%Y-%m-%d')
            partition_dir = output_dir / f"date={date_str}"
            partition_dir.mkdir(exist_ok=True)
            
            output_file = partition_dir / "data.parquet"
            
            # Append if exists, else create
            if output_file.exists():
                existing = pd.read_parquet(output_file)
                combined = pd.concat([existing, group], ignore_index=True)
                combined.to_parquet(output_file, index=False)
            else:
                group.to_parquet(output_file, index=False)
            
            dates_written.add(date_str)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{num_row_groups} row groups")
        
        # Free memory
        del df, table
        import gc
        gc.collect()
    
    logger.info(f"✅ Migrated {len(dates_written)} date partitions to {output_dir}")
    return True


def migrate_silver_data():
    """Convert Silver data to date-partitioned structure"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("MIGRATING SILVER DATA TO PARTITIONED FORMAT")
    logger.info("=" * 70)
    
    # Find source data
    source_paths = [
        SILVER_DIR / 'enriched_lakehouse',
        SILVER_DIR / 'enriched_stocks.parquet',
    ]
    
    source_df = None
    source_path = None
    
    for path in source_paths:
        if path.exists():
            if path.is_dir():
                parquet_files = list(path.glob('*.parquet'))
                if parquet_files:
                    source_df = pd.read_parquet(parquet_files[0])
                    source_path = parquet_files[0]
                    break
            else:
                source_df = pd.read_parquet(path)
                source_path = path
                break
    
    if source_df is None:
        logger.warning("No Silver data found, skipping migration")
        return True
    
    logger.info(f"Source: {source_path}")
    logger.info(f"Rows: {len(source_df):,}")
    
    # Normalize column names
    source_df.columns = [c.lower() for c in source_df.columns]
    
    if 'date' not in source_df.columns:
        logger.error("No 'date' column in Silver data!")
        return False
    
    source_df['date'] = pd.to_datetime(source_df['date'])
    
    # Create output directory
    output_dir = SILVER_DIR / 'enriched_partitioned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique dates
    unique_dates = source_df['date'].dt.date.unique()
    logger.info(f"Unique dates: {len(unique_dates)}")
    
    # Write partitioned data
    logger.info("Writing partitioned files...")
    
    for i, date in enumerate(sorted(unique_dates), 1):
        date_str = date.strftime('%Y-%m-%d')
        partition_dir = output_dir / f"date={date_str}"
        partition_dir.mkdir(exist_ok=True)
        
        mask = source_df['date'].dt.date == date
        df_date = source_df[mask].copy()
        
        output_file = partition_dir / "data.parquet"
        df_date.to_parquet(output_file, index=False)
        
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(unique_dates)} dates")
    
    logger.info(f"✅ Migrated {len(unique_dates)} date partitions to {output_dir}")
    return True


def save_metadata():
    """Save migration metadata"""
    metadata_dir = DATA_DIR / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    
    metadata = {
        'migrated_at': datetime.now().isoformat(),
        'format': 'date-partitioned',
        'partition_column': 'date',
        'partition_format': 'date=YYYY-MM-DD'
    }
    
    with open(metadata_dir / 'partition_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved metadata to {metadata_dir}")


def main():
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATA MIGRATION: CONVERTING TO DATE-PARTITIONED FORMAT")
    logger.info("=" * 70)
    logger.info("")
    
    start_time = datetime.now()
    
    # Migrate Bronze
    bronze_ok = migrate_bronze_prices()
    
    # Migrate Silver
    silver_ok = migrate_silver_data()
    
    # Save metadata
    save_metadata()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Bronze: {'✅' if bronze_ok else '❌'}")
    logger.info(f"  Silver: {'✅' if silver_ok else '❌'}")
    logger.info(f"  Duration: {duration:.1f} seconds")
    logger.info("")
    logger.info("New data locations:")
    logger.info(f"  Bronze: {BRONZE_DIR / 'prices_partitioned'}")
    logger.info(f"  Silver: {SILVER_DIR / 'enriched_partitioned'}")
    logger.info("")
    
    return 0 if (bronze_ok and silver_ok) else 1


if __name__ == "__main__":
    exit(main())
