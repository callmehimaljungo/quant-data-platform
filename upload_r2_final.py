import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Setup Path
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / '.env')

R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')
R2_BUCKET = os.getenv('R2_BUCKET', 'datn')

def get_s3_client():
    return boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        verify=False
    )

def upload_final_data():
    s3 = get_s3_client()
    cache_dir = PROJECT_ROOT / 'data' / 'gold' / 'cache'
    
    mapping = {
        'risk_metrics.parquet': 'processed/gold/cache/risk_metrics.parquet',
        'realtime_metrics.parquet': 'processed/gold/cache/realtime_metrics.parquet',
        'sector_metrics.parquet': 'processed/gold/cache/sector_metrics.parquet',
        'low_beta_quality_weights.parquet': 'processed/gold/cache/low_beta_quality_weights.parquet',
        'sector_rotation_weights.parquet': 'processed/gold/cache/sector_rotation_weights.parquet',
        'momentum_weights.parquet': 'processed/gold/cache/momentum_weights.parquet'
    }
    
    print(f"Syncing local data to R2 bucket: {R2_BUCKET}")
    
    for local_name, remote_key in mapping.items():
        local_path = cache_dir / local_name
        if local_path.exists():
            print(f"Uploading {local_name} -> {remote_key}...")
            s3.upload_file(str(local_path), R2_BUCKET, remote_key)
        else:
            print(f"Skipping {local_name} (not found locally)")

if __name__ == "__main__":
    upload_final_data()
