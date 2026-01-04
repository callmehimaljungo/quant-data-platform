"""
Analyze data coverage across all data types
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR
import pandas as pd

print("📊 DATA COVERAGE ANALYSIS")
print("=" * 80)

# Check Prices
prices_dir = BRONZE_DIR / 'prices_partitioned'
if prices_dir.exists():
    price_partitions = sorted([d.name.replace('date=', '') for d in prices_dir.iterdir() if d.is_dir()])
    price_start = price_partitions[0] if price_partitions else 'N/A'
    price_end = price_partitions[-1] if price_partitions else 'N/A'
    price_count = len(price_partitions)
else:
    price_start = price_end = 'N/A'
    price_count = 0

# Check News
news_dir = BRONZE_DIR / 'market_news_lakehouse'
if news_dir.exists():
    news_files = list(news_dir.glob('*.parquet'))
    if news_files:
        news_df = pd.read_parquet(news_files[0])
        if 'date' in news_df.columns or 'published_at' in news_df.columns:
            date_col = 'date' if 'date' in news_df.columns else 'published_at'
            news_df[date_col] = pd.to_datetime(news_df[date_col])
            news_start = news_df[date_col].min().date()
            news_end = news_df[date_col].max().date()
            news_count = news_df[date_col].nunique()
        else:
            news_start = news_end = 'Unknown'
            news_count = len(news_df)
    else:
        news_start = news_end = 'N/A'
        news_count = 0
else:
    news_start = news_end = 'N/A'
    news_count = 0

# Check Economic
econ_dir = BRONZE_DIR / 'economic_lakehouse'
if econ_dir.exists():
    econ_files = list(econ_dir.glob('*.parquet'))
    if econ_files:
        econ_df = pd.read_parquet(econ_files[0])
        if 'date' in econ_df.columns:
            econ_df['date'] = pd.to_datetime(econ_df['date'])
            econ_start = econ_df['date'].min().date()
            econ_end = econ_df['date'].max().date()
            econ_count = econ_df['date'].nunique()
        else:
            econ_start = econ_end = 'Unknown'
            econ_count = len(econ_df)
    else:
        econ_start = econ_end = 'N/A'
        econ_count = 0
else:
    econ_start = econ_end = 'N/A'
    econ_count = 0

# Create summary table
summary = pd.DataFrame({
    'Data Type': ['Prices', 'News', 'Economic'],
    'Start Date': [price_start, news_start, econ_start],
    'End Date': [price_end, news_end, econ_end],
    'Data Points': [price_count, news_count, econ_count],
    'Status': ['✅ Good', '⚠️ Check', '⚠️ Check']
})

print("\n📋 OVERALL COVERAGE")
print(summary.to_string(index=False))

# Gap analysis for 7 trading days
print("\n\n🔍 GAP ANALYSIS (Dec 23 - Jan 2, 2025)")
print("=" * 80)

trading_days = [
    '2024-12-23',
    '2024-12-24', 
    '2024-12-26',
    '2024-12-27',
    '2024-12-30',
    '2024-12-31',
    '2025-01-02'
]

gap_data = []
for day in trading_days:
    # Check prices
    price_exists = (prices_dir / f'date={day}').exists() if prices_dir.exists() else False
    
    # News & Economic typically don't have daily partitions
    gap_data.append({
        'Date': day,
        'Prices': '✅' if price_exists else '❌',
        'News': '❓',  # Need to check
        'Economic': '❓'  # Need to check
    })

gap_df = pd.DataFrame(gap_data)
print("\n📅 7 TRADING DAYS COVERAGE")
print(gap_df.to_string(index=False))

print("\n\n📊 SUMMARY")
print("=" * 80)
print(f"Prices: {price_count:,} days from {price_start} to {price_end}")
print(f"  Gap: 7 trading days (Dec 23 - Jan 2)")
print(f"  Coverage: {(price_count / (price_count + 7)) * 100:.2f}%")
print(f"\nNews: {news_count:,} records")
print(f"Economic: {econ_count:,} records")
