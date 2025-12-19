"""
Bronze Layer: Ingest raw stock data from Cloudflare R2
Author: Quant Data Platform Team
Date: 2024

Purpose:
- Load raw OHLCV data from R2 storage
- Validate schema according to Section 3.1
- NO transformations (raw data only)
- Add ingestion metadata
- Quality checks: schema validation + null checks on critical columns
"""

import os
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import time

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
    'date': 'datetime64[ns]',
    'ticker': 'object',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64'
}

REQUIRED_COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
CRITICAL_COLUMNS = ['date', 'ticker', 'close']  # Cannot have nulls

OUTPUT_PATH = './data/bronze/prices.parquet'
R2_PATH = 'raw/prices/'


# =============================================================================
# R2 CONNECTION (Section 8.1)
# =============================================================================
def get_r2_client():
    """
    Connect to Cloudflare R2 (S3-compatible storage)
    
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
            if expected_dtype == 'int64' and actual_dtype in ['int32', 'int64']:
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
    
    logger.info("Schema validation PASSED: All checks successful")


# =============================================================================
# DATA INGESTION WITH RETRY LOGIC (Section 7.3)
# =============================================================================
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


def ingest_all_stocks() -> pd.DataFrame:
    """
    Main ingestion function: Load all stock data from R2
    
    Process:
    1. Connect to R2
    2. List all files in raw/prices/
    3. Load each file with retry logic
    4. Concatenate all data
    5. Add ingestion metadata
    6. Validate schema
    
    Returns:
        pd.DataFrame: Consolidated raw data with ingestion timestamp
        
    Raises:
        Exception: If ingestion fails at any step
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BRONZE LAYER INGESTION STARTED")
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
        logger.info(f"Total unique tickers: {df_all['ticker'].nunique():,}")
        logger.info(f"Date range: {df_all['date'].min()} to {df_all['date'].max()}")
        
        # Step 5: Add ingestion metadata
        df_all['ingested_at'] = datetime.now()
        logger.info(f"Added ingestion timestamp: {df_all['ingested_at'].iloc[0]}")
        
        # Step 6: Schema validation
        logger.info("Running schema validation...")
        validate_schema(df_all)
        
        # Success summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 70)
        logger.info(f"BRONZE LAYER INGESTION COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Files processed: {len(dfs)}/{len(parquet_keys)}")
        logger.info(f"Total rows: {len(df_all):,}")
        logger.info(f"Memory usage: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info("=" * 70)
        
        return df_all
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 70)
        logger.error(f"BRONZE LAYER INGESTION FAILED after {duration:.2f} seconds")
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
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size = os.path.getsize(output_path) / 1024**2  # MB
        logger.info(f"Data saved to {output_path}")
        logger.info(f"File size: {file_size:.2f} MB")
        
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
        python bronze/ingest.py
        
    Environment variables required:
        - R2_ENDPOINT
        - R2_ACCESS_KEY
        - R2_SECRET_KEY
        - R2_BUCKET
    """
    try:
        # Ingest data from R2
        df = ingest_all_stocks()
        
        # Save to Bronze layer
        save_to_bronze(df)
        
        logger.info("Bronze layer ingestion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Bronze layer ingestion failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
