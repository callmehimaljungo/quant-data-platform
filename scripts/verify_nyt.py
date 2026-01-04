"""
Verify NYT Archive Collection Results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import BRONZE_DIR

print("=" * 80)
print("📰 NYT ARCHIVE COLLECTION VERIFICATION")
print("=" * 80)

nyt_dir = BRONZE_DIR / 'nyt_archive_lakehouse'

if not nyt_dir.exists():
    print("❌ NYT directory not found!")
    exit(1)

files = sorted(nyt_dir.glob('*.parquet'))
print(f"\n✅ Found {len(files)} parquet files")
print()

total_articles = 0
years_covered = []

for file in files:
    df = pd.read_parquet(file)
    year = file.stem.replace('nyt_', '')
    years_covered.append(year)
    total_articles += len(df)
    
    print(f"  {year}: {len(df):,} articles ({file.stat().st_size / 1024 / 1024:.2f} MB)")

print()
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(f"Total Articles: {total_articles:,}")
print(f"Years Covered: {len(years_covered)} ({min(years_covered)} - {max(years_covered)})")
print(f"Total Size: {sum(f.stat().st_size for f in files) / 1024 / 1024:.2f} MB")
print()

# Sample check
if files:
    sample_df = pd.read_parquet(files[0])
    print("Sample columns:")
    print(f"  {list(sample_df.columns)}")
    print()
    print("Sample article:")
    print(f"  Title: {sample_df.iloc[0]['title']}")
    print(f"  Date: {sample_df.iloc[0]['published_at']}")
    print(f"  Section: {sample_df.iloc[0]['section']}")
