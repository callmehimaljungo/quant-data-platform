import os
import sys
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from dashboard.r2_loader import get_r2_client, get_bucket_name
from config import GOLD_DIR

def upload_to_r2():
    client = get_r2_client()
    if client is None:
        print("❌ R2 Client not configured. Check .env")
        return
    
    bucket = get_bucket_name()
    cache_dir = GOLD_DIR / 'cache'
    
    # Files to sync
    files_to_upload = [
        'risk_metrics.parquet',
        'low_beta_quality_weights.parquet',
        'sector_rotation_weights.parquet',
        'momentum_weights.parquet'
    ]
    
    print(f"🚀 Starting Upload to R2 Bucket: {bucket}")
    
    for filename in files_to_upload:
        local_path = cache_dir / filename
        if not local_path.exists():
            print(f"  ⚠️ Skipping {filename}: Local file not found")
            continue
            
        r2_key = f"processed/gold/cache/{filename}"
        print(f"  ⬆️ Uploading {filename}...")
        
        try:
            with open(local_path, 'rb') as f:
                client.put_object(
                    Bucket=bucket,
                    Key=r2_key,
                    Body=f
                )
            print(f"  ✅ [OK] {filename} uploaded to {r2_key}")
        except Exception as e:
            print(f"  ❌ [ERR] Failed to upload {filename}: {e}")

    print("\n🎉 All verified quality data is now on R2.")

if __name__ == "__main__":
    upload_to_r2()
