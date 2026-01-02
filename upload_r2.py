"""
Upload code/data to Cloudflare R2 Storage
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Optional

from config import R2_CONFIG, R2_PATHS, BRONZE_DIR, SILVER_DIR, GOLD_DIR, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Check boto3
try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Run: pip install boto3")


def get_r2_client():
    """Get R2 client with credentials from environment"""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required for R2 access")
    
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


def upload_file(local_path: Path, r2_key: str) -> bool:
    """Upload a single file to R2"""
    try:
        client = get_r2_client()
        bucket = os.environ['R2_BUCKET']
        
        logger.info(f"Uploading {local_path} -> s3://{bucket}/{r2_key}")
        
        client.upload_file(str(local_path), bucket, r2_key)
        logger.info(f"  [OK] Uploaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"  [FAIL] Upload failed: {e}")
        return False


def upload_directory(local_dir: Path, r2_prefix: str, extensions: List[str] = None) -> int:
    """Upload all files in a directory to R2"""
    if not local_dir.exists():
        logger.error(f"Directory not found: {local_dir}")
        return 0
    
    uploaded = 0
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            # Filter by extension if specified
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            
            # Create R2 key
            relative_path = file_path.relative_to(local_dir)
            r2_key = f"{r2_prefix}/{relative_path}".replace('\\', '/')
            
            if upload_file(file_path, r2_key):
                uploaded += 1
    
    return uploaded


def upload_bronze_to_r2():
    """Upload Bronze layer data to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING BRONZE LAYER TO R2")
    logger.info("=" * 60)
    
    # Upload all_stock_data.parquet
    bronze_parquet = BRONZE_DIR / 'all_stock_data.parquet'
    if bronze_parquet.exists():
        logger.info(f"Bronze prices: {bronze_parquet.stat().st_size / 1e9:.2f} GB")
        upload_file(bronze_parquet, 'bronze/all_stock_data.parquet')
    
    # Upload lakehouse tables
    bronze_lakehouses = [
        'prices_lakehouse',
        'benchmarks_lakehouse', 
        'economic_lakehouse',
        'market_news_lakehouse',
    ]
    
    for lakehouse in bronze_lakehouses:
        lakehouse_path = BRONZE_DIR / lakehouse
        if lakehouse_path.exists():
            count = upload_directory(lakehouse_path, f'bronze/{lakehouse}/', extensions=['.parquet', '.json'])
            logger.info(f"  Uploaded {count} files from {lakehouse}")
    
    # Upload raw files
    raw_dirs = ['economic_raw', 'market_news_raw']
    for raw_dir in raw_dirs:
        raw_path = BRONZE_DIR / raw_dir
        if raw_path.exists():
            count = upload_directory(raw_path, f'bronze/{raw_dir}/', extensions=['.csv', '.json'])
            logger.info(f"  Uploaded {count} files from {raw_dir}")
    
    logger.info("Bronze upload complete!")


def upload_code_to_r2():
    """Upload Python code files to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING CODE TO R2")
    logger.info("=" * 60)
    
    code_prefix = 'code/'
    
    # Upload key Python files
    code_files = [
        'config.py',
        'models/causal_model.py',
        'models/feature_engineering.py',
        'bronze/ingest.py',
        'bronze/economic_loader.py',
        'silver/clean.py',
        'silver/join_schemas.py',
        'requirements.txt',
    ]
    
    for file_rel in code_files:
        file_path = PROJECT_ROOT / file_rel
        if file_path.exists():
            r2_key = f"{code_prefix}{file_rel}".replace('\\', '/')
            upload_file(file_path, r2_key)
    
    logger.info("Code upload complete!")


def upload_data_to_r2():
    """Upload processed data (Silver/Gold) to R2"""
    logger.info("=" * 60)
    logger.info("UPLOADING DATA TO R2")
    logger.info("=" * 60)
    
    # Upload Silver layer parquet
    silver_parquet = SILVER_DIR / 'enriched_stocks.parquet'
    if silver_parquet.exists():
        logger.info(f"Silver data: {silver_parquet.stat().st_size / 1e9:.2f} GB")
        upload_file(silver_parquet, 'processed/silver/enriched_stocks.parquet')
    
    # Upload lakehouse tables
    lakehouses = [
        (SILVER_DIR / 'economic_lakehouse', 'processed/silver/economic_lakehouse/'),
        (SILVER_DIR / 'news_lakehouse', 'processed/silver/news_lakehouse/'),
        (GOLD_DIR, 'processed/gold/'),
    ]
    
    for local_path, r2_prefix in lakehouses:
        if local_path.exists():
            count = upload_directory(local_path, r2_prefix, extensions=['.parquet', '.json'])
            logger.info(f"  Uploaded {count} files from {local_path.name}")
    
    logger.info("Data upload complete!")


def main():
    """Main entry point"""
    import sys
    
    print("=" * 60)
    print("R2 UPLOADER - QUANT DATA PLATFORM")
    print("=" * 60)
    
    # Check environment
    required = ['R2_ENDPOINT', 'R2_ACCESS_KEY', 'R2_SECRET_KEY', 'R2_BUCKET']
    missing = [v for v in required if not os.environ.get(v)]
    
    if missing:
        print(f"\n[ERROR] Missing environment variables: {missing}")
        print("\nPlease set:")
        for var in missing:
            print(f"  set {var}=your_value")
        print("\nOr create a .env file with these values.")
        return 1
    
    print(f"\n[OK] R2 Endpoint: {os.environ['R2_ENDPOINT'][:50]}...")
    print(f"[OK] R2 Bucket: {os.environ['R2_BUCKET']}")
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--code':
            upload_code_to_r2()
        elif sys.argv[1] == '--bronze':
            upload_bronze_to_r2()
        elif sys.argv[1] == '--data':
            upload_data_to_r2()
        elif sys.argv[1] == '--all':
            upload_code_to_r2()
            upload_data_to_r2()
        elif sys.argv[1] == '--full':
            upload_bronze_to_r2()
            upload_data_to_r2()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python upload_r2.py [--code|--bronze|--data|--all|--full]")
            return 1
    else:
        # Default: upload code only
        upload_code_to_r2()
    
    print("\n[DONE] Upload complete!")
    return 0


if __name__ == "__main__":
    exit(main())
