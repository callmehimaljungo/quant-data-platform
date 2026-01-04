"""
Collect 2025 Price Data
Fill gap from 2024-12-23 to present using YFinance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import BRONZE_DIR

print("=" * 80)
print("📈 COLLECTING 2025 PRICE DATA")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()

# Date range to fill
START_DATE = "2024-12-20"  # Last date in Kaggle data
END_DATE = datetime.now().strftime("%Y-%m-%d")

print(f"Date range: {START_DATE} → {END_DATE}")
print()

# Get ticker list from existing Bronze data
prices_dir = BRONZE_DIR / 'prices_partitioned'
if prices_dir.exists():
    partitions = list(prices_dir.iterdir())
    if partitions:
        sample_partition = partitions[-1]
        sample_files = list(sample_partition.glob('*.parquet'))
        if sample_files:
            sample_df = pd.read_parquet(sample_files[0])
            if 'ticker' in sample_df.columns:
                all_tickers = sample_df['ticker'].unique().tolist()[:100]  # Limit to 100 for speed
            else:
                all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
        else:
            all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
    else:
        all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
else:
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']

print(f"Tickers to fetch: {len(all_tickers)}")
print()

# Fetch data using YFinance
print("Fetching data from YFinance...")
all_data = []
errors = 0

for i, ticker in enumerate(all_tickers, 1):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE, end=END_DATE)
        
        if not df.empty:
            df = df.reset_index()
            df['ticker'] = ticker
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'date': 'date'})
            all_data.append(df)
            
        if i % 20 == 0:
            print(f"   Processed {i}/{len(all_tickers)} tickers...")
            
    except Exception as e:
        errors += 1
        continue

print()

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure date column is datetime
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Remove timezone info if present
    if combined_df['date'].dt.tz is not None:
        combined_df['date'] = combined_df['date'].dt.tz_localize(None)
    
    print(f"✅ Fetched {len(combined_df):,} records for {combined_df['ticker'].nunique()} tickers")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Save to Bronze partitions
    print("\nSaving to Bronze partitions...")
    output_dir = BRONZE_DIR / 'prices_partitioned'
    
    for date, group in combined_df.groupby(combined_df['date'].dt.date):
        date_str = str(date)
        partition_dir = output_dir / f'date={date_str}'
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = partition_dir / 'data.parquet'
        group.to_parquet(output_file, index=False)
    
    unique_dates = combined_df['date'].dt.date.nunique()
    print(f"✅ Created {unique_dates} new date partitions")
    
else:
    print("❌ No data fetched!")

print()
print("=" * 80)
print("✅ 2025 PRICE DATA COLLECTION COMPLETE!")
print("=" * 80)
print(f"Errors: {errors}")
