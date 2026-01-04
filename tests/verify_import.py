"""Verify Kaggle import quality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import BRONZE_DIR

# Check imported partitions
partition_dir = BRONZE_DIR / 'prices_partitioned'

# Sample check: 2024-12-20 (latest from Kaggle)
latest_partition = partition_dir / 'date=2024-12-20' / 'data.parquet'

if latest_partition.exists():
    df = pd.read_parquet(latest_partition)
    print(f"✅ Latest Kaggle partition (2024-12-20):")
    print(f"   Rows: {len(df):,}")
    print(f"   Tickers: {df['ticker'].nunique()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample data:")
    print(df.head(3)[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
else:
    print("❌ Latest partition not found!")

# Check gap
import os
all_partitions = sorted([d.name for d in partition_dir.iterdir() if d.is_dir() and d.name.startswith('date=2024-')])
print(f"\n📊 Nov-Dec 2024 partitions: {len([p for p in all_partitions if '2024-11' in p or '2024-12' in p])}")
print(f"   Latest: {all_partitions[-1] if all_partitions else 'None'}")

# Check what's missing
from datetime import datetime, timedelta
target_end = datetime(2025, 1, 2)
current_end = datetime.strptime(all_partitions[-1].replace('date=', ''), '%Y-%m-%d')
gap_days = (target_end - current_end).days

print(f"\n⏳ Remaining gap: {current_end.date()} → {target_end.date()} ({gap_days} days)")
