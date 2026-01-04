"""
Verify Economic Data Collection
"""
import pandas as pd
from pathlib import Path

file_path = Path('data/bronze/economic_lakehouse/economic_indicators.parquet')

print("📊 Economic Data Verification")
print("=" * 70)

df = pd.read_parquet(file_path)

print(f"\n✅ Total Records: {len(df):,}")
print(f"✅ Indicators: {df['indicator'].nunique()}")
print(f"✅ Date Range: {df['date'].min()} to {df['date'].max()}")

print(f"\n📋 Breakdown by Indicator:")
print(df['indicator'].value_counts().to_string())

print(f"\n💾 File Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
print("=" * 70)
