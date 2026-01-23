import pandas as pd
import numpy as np
from pathlib import Path

# Load Gold data
gold_path = Path('data/gold/cache/risk_metrics.parquet')
df = pd.read_parquet(gold_path)
print(f"Original: {len(df)} tickers")
print(f"Columns: {df.columns.tolist()}")

# Check issues
print(f"\n=== Before Cleaning ===")
print(f"sharpe_ratio: min={df['sharpe_ratio'].min():.2f}, max={df['sharpe_ratio'].max():.2f}")
print(f"volatility: min={df['volatility'].min():.2f}, max={df['volatility'].max():.2f}")
print(f"max_drawdown: min={df['max_drawdown'].min():.2f}, max={df['max_drawdown'].max():.2f}")
print(f"NaN sharpe_ratio: {df['sharpe_ratio'].isna().sum()}")
print(f"NaN volatility: {df['volatility'].isna().sum()}")

# Backup original data
backup_path = gold_path.with_name('risk_metrics_raw.parquet')
if not backup_path.exists():
    df.to_parquet(backup_path)
    print(f"✅ Backed up raw data to: {backup_path}")

# Clean data (FILTERING instead of CLIPPING)
print(f"\nScanning for outliers...")
# 1. Fill NaN with median (Keep this)
df['sharpe_ratio'] = df['sharpe_ratio'].fillna(df['sharpe_ratio'].median())
df['volatility'] = df['volatility'].fillna(df['volatility'].median())
df['max_drawdown'] = df['max_drawdown'].fillna(df['max_drawdown'].median())
df['avg_ret'] = df['avg_ret'].fillna(0)

# 2. Filter out garbage data
# Rules: 
# - Max Drawdown < -95% (Almost bankruptcy)
# - Volatility > 400% (High risk/erroneous data)
# - Sharpe < -5 or > 10 (Abnormal)

initial_count = len(df)
df = df[
    (df['max_drawdown'] > -95) & 
    (df['volatility'] < 400) &
    (df['sharpe_ratio'].between(-10, 20))
]
filtered_count = len(df)
dropped_count = initial_count - filtered_count

print(f"🗑️ Dropped {dropped_count:,} tickers ({dropped_count/initial_count:.1%} of data)")
print(f"   - Rules: MaxDD <= -95% OR Vol > 400%")

print(f"\n=== After Cleaning ===")
print(f"sharpe_ratio: min={df['sharpe_ratio'].min():.2f}, max={df['sharpe_ratio'].max():.2f}")
print(f"volatility: min={df['volatility'].min():.2f}, max={df['volatility'].max():.2f}")
print(f"max_drawdown: min={df['max_drawdown'].min():.2f}, max={df['max_drawdown'].max():.2f}")

# Save cleaned data
df.to_parquet(gold_path)
print(f"\n✅ Saved cleaned data: {gold_path}")
print(f"File size: {gold_path.stat().st_size // 1024}KB")
