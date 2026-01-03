"""
R2 Sync Utilities

Download/upload functions for syncing data between local and R2.
Supports all Medallion layers: Bronze, Silver, Gold.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Try to import boto3
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_r2_client():
    """Get R2 client with credentials from environment"""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required. Install with: pip install boto3")
    
    required = ['R2_ENDPOINT', 'R2_ACCESS_KEY', 'R2_SECRET_KEY', 'R2_BUCKET']
    missing = [v for v in required if not os.environ.get(v)]
    
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")
    
    return boto3.client(
        's3',
        endpoint_url=os.environ['R2_ENDPOINT'],
        aws_access_key_id=os.environ['R2_ACCESS_KEY'],
        aws_secret_access_key=os.environ['R2_SECRET_KEY'],
        config=Config(signature_version='s3v4')
    )


def get_bucket():
    """Get R2 bucket name"""
    return os.environ.get('R2_BUCKET', 'datn')


def is_r2_configured() -> bool:
    """Check if R2 credentials are configured"""
    required = ['R2_ENDPOINT', 'R2_ACCESS_KEY', 'R2_SECRET_KEY', 'R2_BUCKET']
    return all(os.environ.get(v) for v in required)


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file_from_r2(r2_key: str, local_path: Path) -> bool:
    """
    Download a single file from R2 to local path.
    
    Args:
        r2_key: Key in R2 bucket (e.g., 'bronze/all_stock_data.parquet')
        local_path: Local path to save file
        
    Returns:
        True if successful
    """
    try:
        client = get_r2_client()
        bucket = get_bucket()
        
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading s3://{bucket}/{r2_key} → {local_path}")
        client.download_file(bucket, r2_key, str(local_path))
        
        size_mb = local_path.stat().st_size / 1024**2
        logger.info(f"  [OK] Downloaded {size_mb:.1f} MB")
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"  [SKIP] File not found in R2: {r2_key}")
        else:
            logger.error(f"  [FAIL] Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Download error: {e}")
        return False


def list_r2_objects(prefix: str) -> List[str]:
    """List all objects in R2 with given prefix"""
    try:
        client = get_r2_client()
        bucket = get_bucket()
        
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        keys = []
        for obj in response.get('Contents', []):
            keys.append(obj['Key'])
        
        return keys
    except Exception as e:
        logger.error(f"Failed to list R2 objects: {e}")
        return []


def download_directory_from_r2(r2_prefix: str, local_dir: Path, 
                                extensions: List[str] = None) -> int:
    """
    Download all files from R2 prefix to local directory.
    
    Args:
        r2_prefix: Prefix in R2 (e.g., 'processed/silver/')
        local_dir: Local directory to save files
        extensions: Optional list of file extensions to filter
        
    Returns:
        Number of files downloaded
    """
    keys = list_r2_objects(r2_prefix)
    
    if not keys:
        logger.warning(f"No files found in R2 prefix: {r2_prefix}")
        return 0
    
    downloaded = 0
    for key in keys:
        # Filter by extension
        if extensions:
            ext = Path(key).suffix.lower()
            if ext not in extensions:
                continue
        
        # Calculate local path
        relative = key[len(r2_prefix):]  # Remove prefix
        local_path = local_dir / relative
        
        if download_file_from_r2(key, local_path):
            downloaded += 1
    
    return downloaded


def download_bronze_from_r2(bronze_dir: Path) -> bool:
    """
    Download Bronze layer data from R2.
    
    Downloads:
    - all_stock_data.parquet (main prices file)
    - *_lakehouse/ directories
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING BRONZE FROM R2")
    logger.info("=" * 60)
    
    success = True
    
    # 1. Download main parquet file
    main_file = bronze_dir / 'all_stock_data.parquet'
    if not download_file_from_r2('bronze/all_stock_data.parquet', main_file):
        success = False
    
    # 2. Download lakehouse directories
    lakehouses = [
        'benchmarks_lakehouse',
        'economic_lakehouse',
        'market_news_lakehouse',
        'stock_metadata_lakehouse',
    ]
    
    for lakehouse in lakehouses:
        local_path = bronze_dir / lakehouse
        r2_prefix = f'bronze/{lakehouse}/'
        count = download_directory_from_r2(r2_prefix, local_path, 
                                            extensions=['.parquet', '.json'])
        logger.info(f"  Downloaded {count} files to {lakehouse}")
    
    logger.info(f"Bronze download complete: {'SUCCESS' if success else 'PARTIAL'}")
    return success


def download_silver_from_r2(silver_dir: Path) -> bool:
    """Download Silver layer data from R2"""
    logger.info("=" * 60)
    logger.info("DOWNLOADING SILVER FROM R2")
    logger.info("=" * 60)
    
    success = True
    
    # 1. Main enriched parquet
    main_file = silver_dir / 'enriched_stocks.parquet'
    if not download_file_from_r2('processed/silver/enriched_stocks.parquet', main_file):
        success = False
    
    # 2. Lakehouse directories
    lakehouses = ['economic_lakehouse', 'news_lakehouse', 'enriched_lakehouse']
    
    for lakehouse in lakehouses:
        local_path = silver_dir / lakehouse
        r2_prefix = f'processed/silver/{lakehouse}/'
        count = download_directory_from_r2(r2_prefix, local_path,
                                            extensions=['.parquet', '.json'])
        if count > 0:
            logger.info(f"  Downloaded {count} files to {lakehouse}")
    
    logger.info(f"Silver download complete: {'SUCCESS' if success else 'PARTIAL'}")
    return success


def download_gold_from_r2(gold_dir: Path) -> bool:
    """Download Gold layer data from R2"""
    logger.info("=" * 60)
    logger.info("DOWNLOADING GOLD FROM R2")
    logger.info("=" * 60)
    
    # Download all parquet and json files
    count = download_directory_from_r2('processed/gold/', gold_dir,
                                        extensions=['.parquet', '.json'])
    
    logger.info(f"Gold download complete: {count} files")
    return count > 0


# =============================================================================
# UPLOAD FUNCTIONS
# =============================================================================

def upload_file_to_r2(local_path: Path, r2_key: str) -> bool:
    """Upload a single file to R2"""
    try:
        client = get_r2_client()
        bucket = get_bucket()
        
        if not local_path.exists():
            logger.warning(f"  [SKIP] File not found: {local_path}")
            return False
        
        logger.info(f"Uploading {local_path} → s3://{bucket}/{r2_key}")
        client.upload_file(str(local_path), bucket, r2_key)
        
        size_mb = local_path.stat().st_size / 1024**2
        logger.info(f"  [OK] Uploaded {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"  [FAIL] Upload failed: {e}")
        return False


def upload_directory_to_r2(local_dir: Path, r2_prefix: str,
                           extensions: List[str] = None) -> int:
    """Upload directory contents to R2"""
    if not local_dir.exists():
        return 0
    
    uploaded = 0
    for file_path in local_dir.rglob('*'):
        if not file_path.is_file():
            continue
        
        # Filter by extension
        if extensions and file_path.suffix.lower() not in extensions:
            continue
        
        relative = file_path.relative_to(local_dir)
        r2_key = f"{r2_prefix}/{relative}".replace('\\', '/')
        
        if upload_file_to_r2(file_path, r2_key):
            uploaded += 1
    
    return uploaded


def upload_bronze_to_r2(bronze_dir: Path) -> bool:
    """Upload Bronze layer to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING BRONZE TO R2")
    logger.info("=" * 60)
    
    success = True
    
    # 1. Main parquet
    main_file = bronze_dir / 'all_stock_data.parquet'
    if main_file.exists():
        if not upload_file_to_r2(main_file, 'bronze/all_stock_data.parquet'):
            success = False
    
    # 2. Lakehouse directories
    lakehouses = [
        'prices_lakehouse',
        'benchmarks_lakehouse',
        'economic_lakehouse',
        'market_news_lakehouse',
        'stock_metadata_lakehouse',
    ]
    
    for lakehouse in lakehouses:
        local_path = bronze_dir / lakehouse
        if local_path.exists():
            count = upload_directory_to_r2(local_path, f'bronze/{lakehouse}',
                                           extensions=['.parquet', '.json'])
            logger.info(f"  Uploaded {count} files from {lakehouse}")
    
    logger.info(f"Bronze upload complete: {'SUCCESS' if success else 'PARTIAL'}")
    return success


def upload_silver_to_r2(silver_dir: Path) -> bool:
    """Upload Silver layer to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING SILVER TO R2")
    logger.info("=" * 60)
    
    success = True
    
    # 1. Main parquet
    main_file = silver_dir / 'enriched_stocks.parquet'
    if main_file.exists():
        if not upload_file_to_r2(main_file, 'processed/silver/enriched_stocks.parquet'):
            success = False
    
    # 2. Lakehouse directories
    lakehouses = ['economic_lakehouse', 'news_lakehouse', 'enriched_lakehouse']
    
    for lakehouse in lakehouses:
        local_path = silver_dir / lakehouse
        if local_path.exists():
            count = upload_directory_to_r2(local_path, f'processed/silver/{lakehouse}',
                                           extensions=['.parquet', '.json'])
            if count > 0:
                logger.info(f"  Uploaded {count} files from {lakehouse}")
    
    logger.info(f"Silver upload complete: {'SUCCESS' if success else 'PARTIAL'}")
    return success


def upload_gold_to_r2(gold_dir: Path) -> bool:
    """Upload Gold layer to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING GOLD TO R2")
    logger.info("=" * 60)
    
    count = upload_directory_to_r2(gold_dir, 'processed/gold',
                                   extensions=['.parquet', '.json'])
    
    logger.info(f"Gold upload complete: {count} files")
    return count > 0


# =============================================================================
# SYNC FUNCTIONS
# =============================================================================

def sync_all_from_r2(data_dir: Path) -> Dict[str, bool]:
    """
    Download all layers from R2.
    
    Args:
        data_dir: Base data directory (contains bronze/, silver/, gold/)
        
    Returns:
        Dict with layer: success status
    """
    results = {}
    
    results['bronze'] = download_bronze_from_r2(data_dir / 'bronze')
    results['silver'] = download_silver_from_r2(data_dir / 'silver')
    results['gold'] = download_gold_from_r2(data_dir / 'gold')
    
    return results


def sync_all_to_r2(data_dir: Path) -> Dict[str, bool]:
    """
    Upload all layers to R2.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dict with layer: success status
    """
    results = {}
    
    results['bronze'] = upload_bronze_to_r2(data_dir / 'bronze')
    results['silver'] = upload_silver_to_r2(data_dir / 'silver')
    results['gold'] = upload_gold_to_r2(data_dir / 'gold')
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for R2 sync operations"""
    import sys
    from pathlib import Path
    
    # Get data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    print("=" * 60)
    print("R2 SYNC UTILITY")
    print("=" * 60)
    
    if not is_r2_configured():
        print("\n[ERROR] R2 not configured. Set environment variables:")
        print("  R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET")
        return 1
    
    if len(sys.argv) < 2:
        print("\nUsage: python r2_sync.py [command]")
        print("\nCommands:")
        print("  download-bronze  Download Bronze layer from R2")
        print("  download-silver  Download Silver layer from R2")
        print("  download-gold    Download Gold layer from R2")
        print("  download-all     Download all layers from R2")
        print("")
        print("  upload-bronze    Upload Bronze layer to R2")
        print("  upload-silver    Upload Silver layer to R2")
        print("  upload-gold      Upload Gold layer to R2")
        print("  upload-all       Upload all layers to R2")
        return 0
    
    command = sys.argv[1].lower()
    
    if command == 'download-bronze':
        download_bronze_from_r2(data_dir / 'bronze')
    elif command == 'download-silver':
        download_silver_from_r2(data_dir / 'silver')
    elif command == 'download-gold':
        download_gold_from_r2(data_dir / 'gold')
    elif command == 'download-all':
        sync_all_from_r2(data_dir)
    elif command == 'upload-bronze':
        upload_bronze_to_r2(data_dir / 'bronze')
    elif command == 'upload-silver':
        upload_silver_to_r2(data_dir / 'silver')
    elif command == 'upload-gold':
        upload_gold_to_r2(data_dir / 'gold')
    elif command == 'upload-all':
        sync_all_to_r2(data_dir)
    else:
        print(f"Unknown command: {command}")
        return 1
    
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    exit(main())
