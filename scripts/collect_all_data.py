"""
Master Data Collection Script
Fetch data from all configured sources
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys
exec(open('config_api_keys.py').read())

from datetime import datetime
import pandas as pd
from config import BRONZE_DIR

print("=" * 80)
print("🚀 MASTER DATA COLLECTION")
print("=" * 80)

# 1. FRED Economic Data
print("\n📊 1. Fetching Economic Data (FRED)...")
try:
    from bronze.collectors.economic.fred_collector import FREDCollector
    
    fred = FREDCollector()
    econ_data = fred.fetch_all(start_date='1962-01-01')
    
    if econ_data:
        output_dir = BRONZE_DIR / 'economic_lakehouse'
        output_file = fred.save_to_bronze(econ_data, output_dir)
        print(f"   ✅ Saved economic data to {output_file}")
    else:
        print("   ⚠️ No economic data fetched")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. NewsAPI Market News
print("\n📰 2. Fetching Market News (NewsAPI)...")
try:
    from bronze.collectors.news.newsapi_collector import NewsAPICollector
    
    newsapi = NewsAPICollector()
    news_df = newsapi.fetch_market_news(days_back=30)
    
    if not news_df.empty:
        output_dir = BRONZE_DIR / 'newsapi_lakehouse'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        news_df.to_parquet(output_file, index=False)
        
        print(f"   ✅ Saved {len(news_df)} articles to {output_file}")
    else:
        print("   ⚠️ No news fetched")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Finnhub News (Alternative)
print("\n📡 3. Fetching Finnhub News...")
try:
    import requests
    import os
    
    url = f"https://finnhub.io/api/v1/news?category=general&token={os.environ['FINNHUB_API_KEY']}"
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        finnhub_news = response.json()
        print(f"   ✅ Got {len(finnhub_news)} Finnhub articles")
        
        # Save
        import json
        output_dir = BRONZE_DIR / 'finnhub_news_lakehouse'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(finnhub_news, f, indent=2)
        
        print(f"   ✅ Saved to {output_file}")
    else:
        print(f"   ⚠️ Finnhub returned {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ MASTER DATA COLLECTION COMPLETE!")
print("=" * 80)

# Summary
print("\n📋 Summary:")
print(f"   Economic Data: {BRONZE_DIR / 'economic_lakehouse'}")
print(f"   NewsAPI: {BRONZE_DIR / 'newsapi_lakehouse'}")
print(f"   Finnhub News: {BRONZE_DIR / 'finnhub_news_lakehouse'}")
print(f"   Prices (existing): {BRONZE_DIR / 'prices_partitioned'}")
