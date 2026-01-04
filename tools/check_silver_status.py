"""
Silver Layer Status Check
Kiểm tra trạng thái và chất lượng dữ liệu Silver layer
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SILVER_DIR
from utils.lakehouse_helper import lakehouse_to_pandas, is_lakehouse_table

print("=" * 70)
print("SILVER LAYER STATUS CHECK")
print("=" * 70)
print()

# 1. Check enriched_stocks.parquet
print("1. ENRICHED STOCKS (Price Data)")
print("-" * 70)
try:
    enriched_path = SILVER_DIR / 'enriched_stocks.parquet'
    df_prices = pd.read_parquet(enriched_path)
    
    print(f"[OK] File: {enriched_path}")
    print(f"[OK] Rows: {len(df_prices):,}")
    print(f"[OK] Tickers: {df_prices['ticker'].nunique():,}")
    print(f"[OK] Columns ({len(df_prices.columns)}): {list(df_prices.columns)}")
    print(f"[OK] Date range: {df_prices['date'].min()} to {df_prices['date'].max()}")
    print(f"[OK] Memory: {df_prices.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"[OK] Sectors: {df_prices['sector'].nunique()}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# 2. Check economic_lakehouse
print("2. ECONOMIC INDICATORS")
print("-" * 70)
try:
    econ_path = SILVER_DIR / 'economic_lakehouse'
    if is_lakehouse_table(econ_path):
        df_econ = lakehouse_to_pandas(econ_path)
        print(f"[OK] Path: {econ_path}")
        print(f"[OK] Rows: {len(df_econ):,}")
        print(f"[OK] Columns ({len(df_econ.columns)}): {list(df_econ.columns)}")
        print(f"[OK] Date range: {df_econ['date'].min()} to {df_econ['date'].max()}")
    else:
        print(f"[NOT FOUND] {econ_path}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# 3. Check news_lakehouse
print("3. NEWS SENTIMENT")
print("-" * 70)
try:
    news_path = SILVER_DIR / 'news_lakehouse'
    if is_lakehouse_table(news_path):
        df_news = lakehouse_to_pandas(news_path)
        print(f"[OK] Path: {news_path}")
        print(f"[OK] Rows: {len(df_news):,}")
        print(f"[OK] Tickers: {df_news['ticker'].nunique() if 'ticker' in df_news.columns else 'N/A'}")
        print(f"[OK] Columns ({len(df_news.columns)}): {list(df_news.columns)}")
        print(f"[OK] Date range: {df_news['date'].min()} to {df_news['date'].max()}")
    else:
        print(f"[NOT FOUND] {news_path}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# 4. Check unified_lakehouse
print("4. UNIFIED DATASET (All Joins)")
print("-" * 70)
try:
    unified_path = SILVER_DIR / 'unified_lakehouse'
    if is_lakehouse_table(unified_path):
        df_unified = lakehouse_to_pandas(unified_path)
        print(f"[OK] Path: {unified_path}")
        print(f"[OK] Rows: {len(df_unified):,}")
        print(f"[OK] Tickers: {df_unified['ticker'].nunique():,}")
        print(f"[OK] Columns ({len(df_unified.columns)}): {list(df_unified.columns)[:10]}...")
        print(f"[OK] Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        print(f"[OK] Memory: {df_unified.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check join coverage
        print("\n  Join Coverage:")
        if 'spy_return' in df_unified.columns:
            spy_coverage = df_unified['spy_return'].notna().sum() / len(df_unified) * 100
            print(f"  - SPY benchmark: {spy_coverage:.1f}%")
        
        if 'vix' in df_unified.columns:
            vix_coverage = df_unified['vix'].notna().sum() / len(df_unified) * 100
            print(f"  - Economic (VIX): {vix_coverage:.1f}%")
        
        if 'daily_sentiment' in df_unified.columns:
            news_coverage = df_unified['daily_sentiment'].notna().sum() / len(df_unified) * 100
            print(f"  - News sentiment: {news_coverage:.1f}%")
    else:
        print(f"[NOT FOUND] {unified_path}")
except Exception as e:
    print(f"[ERROR] {e}")

print()
print("=" * 70)
print("[COMPLETED] STATUS CHECK COMPLETED")
print("=" * 70)
