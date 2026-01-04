"""
Fetch 7 trading days using Twelve Data API
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
from datetime import datetime

os.environ['TWELVE_DATA_API_KEY'] = 'ee3338ef961b48cca43957dce6a7bed9'

from bronze.collectors.prices.twelvedata_collector import TwelveDataCollector
from config import BRONZE_DIR

# 7 trading days to fetch
trading_days = [
    '2024-12-23',  # Monday
    '2024-12-24',  # Tuesday (early close)
    '2024-12-26',  # Thursday
    '2024-12-27',  # Friday
    '2024-12-30',  # Monday
    '2024-12-31',  # Tuesday
    '2025-01-02',  # Thursday
]

print("🚀 Fetching 7 Trading Days with Twelve Data")
print(f"   API Key: {os.environ['TWELVE_DATA_API_KEY'][:10]}...")
print(f"   Days: {len(trading_days)}")

# Get S&P 500 tickers from Kaggle
kaggle_df = pd.read_csv('data/temp/sp500_stocks.csv', nrows=1000)
sp500_tickers = sorted(kaggle_df['Symbol'].unique().tolist())[:100]  # Start with 100 tickers
print(f"   Tickers: {len(sp500_tickers)} (batch 1)")

collector = TwelveDataCollector()

if not collector.is_available():
    print("❌ API key not available!")
    exit(1)

print("✅ Twelve Data collector ready\n")

total_rows = 0
for day in trading_days:
    print(f"📅 Fetching {day}...")
    
    try:
        # Fetch for this day
        df = collector.fetch(sp500_tickers, day, day)
        
        if df.empty:
            print(f"   ⚠️ No data returned")
            continue
        
        # Add ingested_at
        df['ingested_at'] = datetime.now()
        
        # Save to partition
        date_obj = pd.to_datetime(day).date()
        date_str = date_obj.strftime('%Y-%m-%d')
        partition_dir = BRONZE_DIR / 'prices_partitioned' / f"date={date_str}"
        partition_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = partition_dir / "data.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"   ✅ Saved {len(df)} rows to {date_str}")
        total_rows += len(df)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n🎉 DONE!")
print(f"   Total rows: {total_rows:,}")
print(f"   Days completed: {total_rows // len(sp500_tickers) if sp500_tickers else 0}/{len(trading_days)}")
