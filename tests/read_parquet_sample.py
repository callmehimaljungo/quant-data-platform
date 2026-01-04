"""
Read and display content of a specific parquet file
"""
import pandas as pd
from pathlib import Path

file_path = Path(r"e:\GitHub\quant-trade\quant-data-platform\data\bronze\market_news_lakehouse\data_v0_20251230_092103.parquet")

print(f"📂 Reading file: {file_path}")

try:
    df = pd.read_parquet(file_path)
    
    print(f"\n📊 Metadata:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Column Names: {list(df.columns)}")
    
    print(f"\n📄 Sample Data (First 3 rows):")
    # Adjust options to ensure columns aren't hidden
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(3))
    
    print(f"\n🗓️ Date Range:")
    if 'date' in df.columns:
        print(f"   {df['date'].min()} to {df['date'].max()}")
    elif 'published_at' in df.columns:
        print(f"   {df['published_at'].min()} to {df['published_at'].max()}")
    else:
        print("   No obvious date column found")

except Exception as e:
    print(f"❌ Error reading file: {e}")
