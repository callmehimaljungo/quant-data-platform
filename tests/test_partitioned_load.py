"""Test partitioned Bronze loading"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from silver.clean import load_bronze_data

# Test loading recent dates only
df = load_bronze_data(start_date='2024-11-01', end_date='2024-11-04')

print(f"✅ Loaded {len(df):,} rows")
print(f"✅ Tickers: {df['ticker'].nunique()}")
print(f"✅ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✅ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
