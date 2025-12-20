"""
Bronze Layer: Ingest raw stock data from Kaggle or Cloudflare R2
Author: Quant Data Platform Team
Date: 2024

Purpose:
- Load raw OHLCV data from Kaggle API or R2 storage
- Validate schema according to Section 3.1
- NO transformations (raw data only)
- Add ingestion metadata
- Quality checks: schema validation + null checks on critical columns

Data Sources:
- PRIMARY: Kaggle (hmingjungo/stock-price dataset)
- ALTERNATIVE: Cloudflare R2 (if data already uploaded)
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

# Try to import optional dependencies
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False
    print("⚠️  boto3 not installed - R2 support disabled")

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️  kaggle not installed - Kaggle support disabled")

# =============================================================================
# LOGGING SETUP (Section 7.2)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (Section 3.1 - Price Data Schema)
# =============================================================================
EXPECTED_SCHEMA = {
    'Date': 'datetime64[ns]',      # Note: Kaggle uses 'Date' not 'date'
    'Ticker': 'object',             # Note: Kaggle uses 'Ticker' not 'ticker'
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'int64'
}

REQUIRED_COLUMNS = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
CRITICAL_COLUMNS = ['Date', 'Ticker', 'Close']  # Cannot have nulls

OUTPUT_PATH = './data/bronze/prices.parquet'
KAGGLE_DATASET = 'hmingjungo/stock-price'
TEMP_DIR = './temp'
R2_PATH = 'raw/prices/'


# =============================================================================
# KAGGLE INGESTION (NEW - PRIMARY METHOD)
# =============================================================================
def ingest_from_kaggle(dataset: str = KAGGLE_DATASET) -> pd.DataFrame:
    """
    Download and load stock data directly from Kaggle
    
    This is the PRIMARY method for Bronze layer ingestion.
    
    Process:
    1. Download dataset from Kaggle API
    2. Extract CSV file
    3. Load into DataFrame
    4. Standardize column names
    5. Validate schema
    
    Args:
        dataset: Kaggle dataset identifier (e.g., 'hmingjungo/stock-price')
        
    Returns:
        pd.DataFrame: Raw stock data
        
    Raises:
        ImportError: If kaggle package not installed
        Exception: If download or loading fails
    """
    if not KAGGLE_AVAILABLE:
        raise ImportError(
            "Kaggle package not installed. Install with: pip install kaggle"
        )
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER INGESTION FROM KAGGLE")
    logger.info("=" * 70)
    
    try:
        # Create temp directory
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Step 1: Download from Kaggle
        logger.info(f"Downloading dataset: {dataset}")
        kaggle.api.dataset_download_files(
            dataset,
            path=TEMP_DIR,
            unzip=True
        )
        logger.info("✓ Download completed")
        
        # Step 2: Find CSV file (Kaggle extracts to various names)
        csv_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {TEMP_DIR}")
        
        # Use the largest CSV (usually all_stock_data.csv or stock_price.csv)
        csv_path = os.path.join(TEMP_DIR, csv_files[0])
        logger.info(f"Loading file: {csv_files[0]}")
        
        # Step 3: Load CSV with proper dtypes
        logger.info("Loading CSV file... (this may take a few minutes)")
        df = pd.read_csv(
            csv_path,
            parse_dates=['Date'],
            dtype={
                'Ticker': 'object',
                'Open': 'float64',
                'High': 'float64',
                'Low': 'float64',
                'Close': 'float64',
                'Volume': 'int64'
            }
        )
        
        logger.info(f"✓ Loaded {len(df):,} rows")
        logger.info(f"✓ Unique tickers: {df['Ticker'].nunique():,}")
        logger.info(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Step 4: Standardize column names (lowercase for consistency)
        # Note: Keep original for now, will standardize in Silver layer
        
        # Clean up temp files
        try:
            os.remove(csv_path)
            logger.info("✓ Cleaned up temp files")
        except:
            pass
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 70)
        logger.info(f"KAGGLE INGESTION COMPLETED")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 70)
        
        return df
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"KAGGLE INGESTION FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


# =============================================================================
# R2 CONNECTION (Section 8.1) - ALTERNATIVE METHOD
# =============================================================================
def get_r2_client():
    """
    Connect to Cloudflare R2 (S3-compatible storage)
    
    This is an ALTERNATIVE method if data is already in R2.
    
    Required environment variables:
    - R2_ENDPOINT: R2 endpoint URL
    - R2_ACCESS_KEY: Access key ID
    - R2_SECRET_KEY: Secret access key
    - R2_BUCKET: Bucket name
    
    Returns:
        boto3.client: S3 client configured for R2
    
    Raises:
        ValueError: If required environment variables are missing
    """
    if not R2_AVAILABLE:
        raise ImportError("boto3 not installed. Install with: pip install boto3")
    
    required_vars = ['R2_ENDPOINT', 'R2_ACCESS_KEY', 'R2_SECRET_KEY', 'R2_BUCKET']
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    try:
        client = boto3.client(
            's3',
            endpoint_url=os.environ['R2_ENDPOINT'],
            aws_access_key_id=os.environ['R2_ACCESS_KEY'],
            aws_secret_access_key=os.environ['R2_SECRET_KEY'],
            config=Config(signature_version='s3v4')
        )
        logger.info("Successfully connected to R2 storage")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to R2: {str(e)}")
        raise


def load_from_r2_with_retry(
    client, 
    bucket: str, 
    key: str, 
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Load a single parquet file from R2 with retry logic
    
    Args:
        client: boto3 S3 client
        bucket: R2 bucket name
        key: Object key/path
        max_retries: Maximum number of retry attempts
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load {key} (attempt {attempt + 1}/{max_retries})")
            
            response = client.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(response['Body'])
            
            logger.info(f"Successfully loaded {key}: {len(df)} rows")
            return df
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File not found: {key}")
                raise
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Failed to load {key}: {str(e)}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to load {key} after {max_retries} attempts")
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error loading {key}: {str(e)}")
            raise


def list_r2_objects(client, bucket: str, prefix: str) -> List[str]:
    """
    List all objects in R2 bucket with given prefix
    
    Args:
        client: boto3 S3 client
        bucket: R2 bucket name
        prefix: Object prefix/path
        
    Returns:
        List of object keys
    """
    try:
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        keys = []
        for page in pages:
            if 'Contents' in page:
                keys.extend([obj['Key'] for obj in page['Contents']])
        
        logger.info(f"Found {len(keys)} objects in {prefix}")
        return keys
        
    except Exception as e:
        logger.error(f"Failed to list objects in {prefix}: {str(e)}")
        raise


def ingest_from_r2() -> pd.DataFrame:
    """
    Load stock data from R2 storage (ALTERNATIVE METHOD)
    
    Use this if you have already uploaded data to R2.
    
    Process:
    1. Connect to R2
    2. List all files in raw/prices/
    3. Load each file with retry logic
    4. Concatenate all data
    
    Returns:
        pd.DataFrame: Consolidated raw data
        
    Raises:
        Exception: If ingestion fails at any step
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER INGESTION FROM R2")
    logger.info("=" * 70)
    
    try:
        # Step 1: Connect to R2
        client = get_r2_client()
        bucket = os.environ['R2_BUCKET']
        
        # Step 2: List all files
        logger.info(f"Listing objects in {R2_PATH}...")
        object_keys = list_r2_objects(client, bucket, R2_PATH)
        
        if not object_keys:
            raise ValueError(f"No files found in {R2_PATH}")
        
        # Filter for parquet files only
        parquet_keys = [k for k in object_keys if k.endswith('.parquet')]
        logger.info(f"Found {len(parquet_keys)} parquet files to process")
        
        # Step 3: Load all files
        dfs = []
        failed_files = []
        
        for i, key in enumerate(parquet_keys, 1):
            try:
                df = load_from_r2_with_retry(client, bucket, key)
                dfs.append(df)
                
                if i % 100 == 0:
                    logger.info(f"Progress: {i}/{len(parquet_keys)} files loaded")
                    
            except Exception as e:
                logger.error(f"Failed to load {key}: {str(e)}")
                failed_files.append(key)
        
        if not dfs:
            raise ValueError("No data was successfully loaded")
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files[:10]}...")
        
        # Step 4: Concatenate all data
        logger.info("Concatenating all data...")
        df_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows loaded: {len(df_all):,}")
        logger.info(f"Total unique tickers: {df_all['Ticker'].nunique():,}")
        logger.info(f"Date range: {df_all['Date'].min()} to {df_all['Date'].max()}")
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 70)
        logger.info(f"R2 INGESTION COMPLETED")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Files processed: {len(dfs)}/{len(parquet_keys)}")
        logger.info("=" * 70)
        
        return df_all
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"R2 INGESTION FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


# =============================================================================
# SCHEMA VALIDATION (Section 7.4)
# =============================================================================
def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate DataFrame schema against expected schema
    
    Checks:
    1. All required columns present
    2. Data types match expected types
    3. No nulls in critical columns
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If schema validation fails
    """
    # Check for missing columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema validation FAILED: Missing columns {missing_cols}")
    
    # Check data types
    type_mismatches = []
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            # Allow some flexibility for numeric types
            if expected_dtype == 'float64' and actual_dtype in ['float32', 'float64']:
                continue
            if expected_dtype == 'int64' and actual_dtype in ['int32', 'int64', 'Int64']:
                continue
            if expected_dtype != actual_dtype:
                type_mismatches.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
    
    if type_mismatches:
        raise ValueError(f"Schema validation FAILED: Type mismatches - {type_mismatches}")
    
    # Check for nulls in critical columns
    null_counts = {}
    for col in CRITICAL_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_counts[col] = null_count
    
    if null_counts:
        raise ValueError(
            f"Schema validation FAILED: Null values in critical columns - {null_counts}"
        )
    
    logger.info("✓ Schema validation PASSED: All checks successful")


# =============================================================================
# UNIFIED INGESTION FUNCTION
# =============================================================================
def ingest_all_stocks(source: str = 'auto') -> pd.DataFrame:
    """
    Main ingestion function with automatic source detection
    
    Args:
        source: Data source ('kaggle', 'r2', or 'auto')
                'auto' will try Kaggle first, then R2
    
    Returns:
        pd.DataFrame: Raw stock data with ingestion timestamp
    """
    start_time = datetime.now()
    
    try:
        # Auto-detect source
        if source == 'auto':
            if KAGGLE_AVAILABLE:
                logger.info("🔍 Auto-detect: Using Kaggle as primary source")
                source = 'kaggle'
            elif R2_AVAILABLE and os.environ.get('R2_ENDPOINT'):
                logger.info("🔍 Auto-detect: Using R2 as fallback source")
                source = 'r2'
            else:
                raise ValueError(
                    "No data source available. Please install either:\n"
                    "  - kaggle: pip install kaggle\n"
                    "  - boto3: pip install boto3 (and configure R2)"
                )
        
        # Ingest from selected source
        if source == 'kaggle':
            df = ingest_from_kaggle()
        elif source == 'r2':
            df = ingest_from_r2()
        else:
            raise ValueError(f"Unknown source: {source}. Use 'kaggle', 'r2', or 'auto'")
        
        # Add ingestion metadata
        df['ingested_at'] = datetime.now()
        logger.info(f"✓ Added ingestion timestamp: {df['ingested_at'].iloc[0]}")
        
        # Validate schema
        logger.info("Running schema validation...")
        validate_schema(df)
        
        # Success summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 70)
        logger.info(f"✓✓✓ BRONZE LAYER INGESTION COMPLETED SUCCESSFULLY ✓✓✓")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info("=" * 70)
        
        return df
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"❌ BRONZE LAYER INGESTION FAILED after {duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise


def save_to_bronze(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """
    Save DataFrame to Bronze layer (Parquet format)
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as Parquet with compression
        logger.info(f"Saving to {output_path}...")
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size = os.path.getsize(output_path) / 1024**2  # MB
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
    Main execution function
    
    Usage:
        python bronze/ingest.py              # Auto-detect source
        python bronze/ingest.py kaggle       # Force Kaggle
        python bronze/ingest.py r2           # Force R2
    
    For Kaggle (default):
        - Requires: pip install kaggle
        - Requires: ~/.kaggle/kaggle.json with API credentials
    
    For R2 (alternative):
        - Requires: pip install boto3
        - Environment variables: R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
    """
    import sys
    
    # Get source from command line argument
    source = sys.argv[1] if len(sys.argv) > 1 else 'auto'
    
    logger.info("")
    logger.info("🚀 BRONZE LAYER INGESTION")
    logger.info(f"📊 Data Source: {source}")
    logger.info("")
    
    try:
        # Ingest data
        df = ingest_all_stocks(source=source)
        
        # Save to Bronze layer
        save_to_bronze(df)
        
        logger.info("")
        logger.info("✅ Bronze layer ingestion completed successfully!")
        logger.info(f"✅ Output: {OUTPUT_PATH}")
        logger.info("")
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Bronze layer ingestion failed: {str(e)}")
        logger.error("")
        return 1


if __name__ == "__main__":
    exit(main())
