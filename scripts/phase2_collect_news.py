"""
Phase 2: News Data Collection
Collect recent news (30 days) from NewsAPI and Finnhub
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys
exec(open('config_api_keys.py').read())

from datetime import datetime
from config import BRONZE_DIR
import pandas as pd

print("=" * 80)
print("📰 PHASE 2: NEWS DATA COLLECTION")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()

# Step 1: NewsAPI (Recent 30 days)
print("Step 1: Fetching recent news from NewsAPI (last 30 days)...")
try:
    from bronze.collectors.news.newsapi_collector import NewsAPICollector
    
    newsapi = NewsAPICollector()
    news_df = newsapi.fetch_market_news(days_back=30, page_size=100)
    
    if not news_df.empty:
        output_dir = BRONZE_DIR / 'newsapi_lakehouse'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        news_df.to_parquet(output_file, index=False)
        
        print(f"   ✅ NewsAPI: {len(news_df)} articles saved to {output_file}")
    else:
        print("   ⚠️ NewsAPI: No articles fetched")
        
except Exception as e:
    print(f"   ❌ NewsAPI Error: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Finnhub News (General market news)
print("\nStep 2: Fetching Finnhub general market news...")
try:
    import requests
    import os
    import json
    
    url = f"https://finnhub.io/api/v1/news?category=general&token={os.environ['FINNHUB_API_KEY']}"
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        finnhub_news = response.json()
        
        if finnhub_news:
            # Convert to DataFrame
            records = []
            for article in finnhub_news:
                records.append({
                    'news_id': article.get('id'),
                    'title': article.get('headline'),
                    'summary': article.get('summary'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'published_at': pd.to_datetime(article.get('datetime'), unit='s'),
                    'category': article.get('category'),
                    'related': article.get('related'),
                    'raw_json': json.dumps(article)
                })
            
            df = pd.DataFrame(records)
            df['fetched_at'] = datetime.now()
            
            output_dir = BRONZE_DIR / 'finnhub_news_lakehouse'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f'news_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
            df.to_parquet(output_file, index=False)
            
            print(f"   ✅ Finnhub: {len(df)} articles saved to {output_file}")
        else:
            print("   ⚠️ Finnhub: No articles returned")
    else:
        print(f"   ⚠️ Finnhub: HTTP {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Finnhub Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ PHASE 2 COMPLETE!")
print("=" * 80)

# Summary
print("\nNews Data Collected:")
newsapi_dir = BRONZE_DIR / 'newsapi_lakehouse'
finnhub_dir = BRONZE_DIR / 'finnhub_news_lakehouse'

newsapi_count = 0
if newsapi_dir.exists():
    for f in newsapi_dir.glob('*.parquet'):
        df = pd.read_parquet(f)
        newsapi_count += len(df)

finnhub_count = 0
if finnhub_dir.exists():
    for f in finnhub_dir.glob('*.parquet'):
        df = pd.read_parquet(f)
        finnhub_count += len(df)

print(f"   NewsAPI: {newsapi_count} articles")
print(f"   Finnhub: {finnhub_count} articles")
print(f"   Total: {newsapi_count + finnhub_count} articles")
