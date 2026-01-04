"""Convert Kaggle S&P 500 data to Bronze partitions"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from config import BRONZE_DIR

# Read Kaggle data
print("Loading Kaggle data...")
df = pd.read_csv('data/temp/sp500_stocks.csv')

# Filter to gap period (2024-11-05 to 2024-12-20)
df['Date'] = pd.to_datetime(df['Date'])
df_gap = df[(df['Date'] >= '2024-11-05') & (df['Date'] <= '2024-12-20')].copy()

print(f"Gap data: {len(df_gap):,} rows")
print(f"Date range: {df_gap['Date'].min().date()} to {df_gap['Date'].max().date()}")
print(f"Tickers: {df_gap['Date'].nunique()}")

# Rename columns to match Bronze schema
df_gap = df_gap.rename(columns={
    'Date': 'date',
    'Symbol': 'ticker',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})

# Add ingested_at
df_gap['ingested_at'] = datetime.now()

# Save to partitions
output_dir = BRONZE_DIR / 'prices_partitioned'
print(f"\nSaving to partitions: {output_dir}")

dates_saved = []
for date, group in df_gap.groupby(df_gap['date'].dt.date):
    date_str = date.strftime('%Y-%m-%d')
    partition_dir = output_dir / f"date={date_str}"
    partition_dir.mkdir(exist_ok=True)
    
    output_file = partition_dir / "data.parquet"
    group.to_parquet(output_file, index=False)
    dates_saved.append(date_str)
    
    if len(dates_saved) % 5 == 0:
        print(f"  Saved {len(dates_saved)} partitions...")

print(f"\n✅ Saved {len(dates_saved)} date partitions")
print(f"✅ Date range: {min(dates_saved)} to {max(dates_saved)}")
