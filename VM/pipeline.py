import os
import sys
import io
import boto3
import duckdb
import pandas as pd
import numpy as np
import logging
import warnings
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# --- CONFIGURATION ---
# The VM path is expected to be /opt/quant, but we use Path(__file__).parent
# to make it portable if run from the checkout directory.
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if '/opt/quant' in str(PROJECT_ROOT):
    BASE_DIR = Path('/opt/quant')
else:
    BASE_DIR = PROJECT_ROOT

DATA_DIR = BASE_DIR / 'data'
BRONZE_DIR = DATA_DIR / 'bronze'
SILVER_DIR = DATA_DIR / 'silver'
GOLD_DIR = DATA_DIR / 'gold'
SILVER_FILE = SILVER_DIR / 'enriched_stocks.parquet'
CACHE_RISK = GOLD_DIR / 'cache' / 'risk_metrics.parquet'
CACHE_SECTOR = GOLD_DIR / 'cache' / 'sector_metrics.parquet'

# Add packages to sys.path for modular imports
sys.path.insert(0, str(BASE_DIR))

# Load Environment Variables
load_dotenv(BASE_DIR / '.env')

R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY')
R2_SECRET_KEY = os.getenv('R2_SECRET_KEY')
R2_BUCKET = os.getenv('R2_BUCKET', 'datn')

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def get_s3_client():
    """Build S3/R2 Client"""
    if not all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY]):
        logger.error("R2 Credentials missing in .env")
        return None
    return boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        verify=False
    )

# --- 1. SYNC METHODS ---

def sync_bronze_data():
    """Download Bronze data from R2"""
    logger.info("=== ☁️ Syncing Bronze Data ===")
    s3 = get_s3_client()
    if not s3: return
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download historical base
    target_hist = BRONZE_DIR / 'all_stock_data.parquet'
    try:
        logger.info("Checking for 'all_stock_data.parquet'...")
        s3.download_file(R2_BUCKET, 'raw/prices/all_stock_data.parquet', str(target_hist))
    except Exception:
        logger.warning("All_stock_data missing on R2, continuing with daily fragments.")

    # 2. Sync daily fragments
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix='raw/prices/')
    dfs = []
    if 'Contents' in resp:
        for o in resp['Contents']:
            if o['Key'].endswith('.parquet') and 'all_stock_data' not in o['Key']:
                r = s3.get_object(Bucket=R2_BUCKET, Key=o['Key'])
                dfs.append(pd.read_parquet(io.BytesIO(r['Body'].read())))
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.columns = [c.lower() for c in merged.columns]
        merged.to_parquet(BRONZE_DIR / 'prices.parquet')
        logger.info(f"Merged {len(dfs)} fragments into prices.parquet.")

def sync_silver_and_gold_from_r2():
    """Pulls existing processed data to avoid full recompute if possible"""
    logger.info("=== ☁️ Syncing Silver/Gold from R2 (Incremental) ===")
    s3 = get_s3_client()
    if not s3: return
    
    # Silver base
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix='processed/silver/')
    if 'Contents' in resp:
        cands = [o for o in resp['Contents'] if o['Key'].endswith('.parquet')]
        if cands:
            target = max(cands, key=lambda x: x['Size'])
            logger.info(f"Syncing Silver: {target['Key']}")
            s3.download_file(R2_BUCKET, target['Key'], str(SILVER_FILE))

# --- 2. PROCESSING METHODS ---

def run_silver_clean():
    """Runs the modular Silver processor"""
    logger.info("=== 🥇 Running Silver Clean (Modular) ===")
    try:
        from silver.run_all_processors import run_all_processors
        run_all_processors()
    except ImportError:
        logger.warning("Modular silver processors not found. Attempting simple fallback...")
        # Fallback to a simple dropna if modular is missing
        if (BRONZE_DIR / 'prices.parquet').exists():
            df = pd.read_parquet(BRONZE_DIR / 'prices.parquet')
            df = df.dropna().drop_duplicates()
            df.to_parquet(SILVER_FILE)

def calculate_gold_optimized():
    """The 'Secret Sauce' - Optimized DuckDB for Risk Metrics"""
    logger.info("=== 🏆 Calculating Gold Metrics (Optimized Mode) ===")
    if not SILVER_FILE.exists():
        logger.error("No Silver data found. Run Silver clean first.")
        return

    try:
        con = duckdb.connect()
        # Fast aggregation
        df = con.execute(f"SELECT ticker, AVG(daily_return) as avg_ret, STDDEV(daily_return) as std_ret, COUNT(*) as count, MAX(date) as last_date FROM read_parquet('{SILVER_FILE}') GROUP BY ticker HAVING count > 50").fetchdf()
        
        # Metrics (Input std_ret is in Percent, e.g. 1.0 for 1%)
        # Annualized Volatility (%) = daily_std (%) * sqrt(252)
        df['volatility'] = (df['std_ret'] * 15.8745)
        
        # Sharpe Ratio = (Ann_Ret / Ann_Vol). Ratio is unitless if both are %
        df['sharpe_ratio'] = (df['avg_ret'] * 252) / (df['volatility'] + 0.1)
        df['sharpe_ratio'] = df['sharpe_ratio'].clip(-5, 5)
        
        # MaxDD (Approx via Window)
        con.execute("CREATE TABLE t_list AS SELECT ticker FROM df")
        df_dd = con.execute(f"WITH p AS (SELECT * FROM read_parquet('{SILVER_FILE}') p SEMI JOIN t_list t ON p.ticker=t.ticker), w AS (SELECT ticker, (close/MAX(close) OVER(PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING))-1.0 as dd FROM p) SELECT ticker, MIN(dd) as max_dd_raw FROM w GROUP BY ticker").fetchdf()
        df = df.merge(df_dd, on='ticker', how='left')
        df['max_drawdown'] = (df['max_dd_raw'] * 100)
        
        # --- STRICT FILTERING ---
        # 1. Volatility < 100% (Annualized standard threshold)
        # 2. Max Drawdown > -80% (Preserve realistic stocks)
        n_before = len(df)
        df = df[(df['volatility'] < 100) & (df['max_drawdown'] > -80)].copy()
        logger.info(f"Filtering: Removed {n_before - len(df)} stocks with excessive risk (Vol > 100% or DD < -80%). Remaining: {len(df)}")
        
        # Clip for visualization stability
        df['volatility'] = df['volatility'].clip(2, 100)
        df['max_drawdown'] = df['max_drawdown'].clip(-99.9, 0)
        
        df['sector'] = 'Unknown'
        df = df.rename(columns={'last_date': 'date'})
        
        CACHE_RISK.parent.mkdir(parents=True, exist_ok=True)
        df[['ticker', 'sector', 'sharpe_ratio', 'volatility', 'max_drawdown', 'date']].to_parquet(CACHE_RISK)
        
        # Sector median
        sec = df.groupby('sector').agg({'ticker':'count', 'sharpe_ratio':'median', 'volatility':'median', 'max_drawdown':'median'}).reset_index()
        sec.columns = ['sector', 'num_tickers', 'sharpe_ratio', 'volatility', 'max_drawdown']
        sec.to_parquet(CACHE_SECTOR)
        logger.info(f"Gold OK: {len(df)} tickers.")
    except Exception as e:
        logger.error(f"Gold processing failed: {e}")

# --- 3. UPLOAD METHODS ---

def upload_to_r2():
    """Uploads locally computed cache to R2"""
    logger.info("=== ☁️ Uploading Cache to R2 ===")
    s3 = get_s3_client()
    if not s3: return
    
    mapping = [
        (CACHE_RISK, 'processed/gold/cache/risk_metrics.parquet'),
        (CACHE_RISK, 'processed/gold/cache/realtime_metrics.parquet'), # For dashboard
        (CACHE_SECTOR, 'processed/gold/cache/sector_metrics.parquet')
    ]
    
    for local, remote in mapping:
        if local.exists():
            s3.upload_file(str(local), R2_BUCKET, remote)
            logger.info(f"Done: {remote}")

# --- ORCHESTRATION ---

def main():
    parser = argparse.ArgumentParser(description="Master VM Pipeline")
    parser.add_argument("--sync-only", action="store_true", help="Only sync from R2")
    parser.add_argument("--process-only", action="store_true", help="Only run local processing")
    parser.add_argument("--upload-only", action="store_true", help="Only upload to R2")
    args = parser.parse_args()

    logger.info(f"--- STARTING PIPELINE ({datetime.now()}) ---")
    
    # Sequence
    if args.sync_only:
        sync_silver_and_gold_from_r2()
        sync_bronze_data()
    elif args.upload_only:
        upload_to_r2()
    else:
        # Full run
        sync_silver_and_gold_from_r2()
        sync_bronze_data()
        run_silver_clean()
        calculate_gold_optimized()
        upload_to_r2()
        
    logger.info("--- PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()
