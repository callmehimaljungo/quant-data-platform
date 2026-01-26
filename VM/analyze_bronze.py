import boto3
import os
import pandas as pd
import io
from dotenv import load_dotenv

load_dotenv('/opt/quant/.env')

s3 = boto3.client('s3',
    endpoint_url=os.environ['R2_ENDPOINT'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY']
)
BUCKET = os.environ['R2_BUCKET']

print("=== Bronze Data Structure ===")

# Get Bronze prices file
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix='raw/prices/')
print("Bronze files on R2:")
total_size = 0
for o in resp.get('Contents', []):
    print(f"  {o['Key']} - {o['Size']//1024}KB")
    total_size += o['Size']
print(f"Total Bronze size: {total_size//1024//1024}MB")

# Check if there's a larger Bronze file
resp2 = s3.list_objects_v2(Bucket=BUCKET, Prefix='processed/bronze/')
print("\nProcessed Bronze files:")
for o in resp2.get('Contents', [])[:10]:
    print(f"  {o['Key']} - {o['Size']//1024}KB")

# Check local Bronze on VM
print("\n=== Local Bronze on VM ===")
import subprocess
result = subprocess.run(['ls', '-la', '/opt/quant/data/bronze/'], capture_output=True, text=True)
print(result.stdout)

# Check local Bronze size
from pathlib import Path
bronze_path = Path('/opt/quant/data/bronze/prices.parquet')
if bronze_path.exists():
    df = pd.read_parquet(bronze_path)
    print(f"\nLocal Bronze: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if 'ticker' in df.columns:
        print(f"Unique tickers: {df['ticker'].nunique()}")
    elif 'Ticker' in df.columns:
        print(f"Unique tickers: {df['Ticker'].nunique()}")
else:
    print("Local Bronze not found!")

# Check if there's all_stock_data.parquet
alt_path = Path('/opt/quant/data/bronze/all_stock_data.parquet')
if alt_path.exists():
    df_alt = pd.read_parquet(alt_path)
    print(f"\nalt file: {len(df_alt)} rows")
    print(f"Columns: {df_alt.columns.tolist()}")
