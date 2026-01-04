"""
Phase 3: Historical News Collection (2020-2024)
Batch collect from Finnhub with rate limiting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys
exec(open('config_api_keys.py').read())

from datetime import datetime, timedelta
from config import BRONZE_DIR
import pandas as pd
import requests
import os
import json
import time

print("=" * 80)
print("📚 PHASE 3: HISTORICAL NEWS COLLECTION (2020-2024)")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print()

# Finnhub rate limit: 60 req/min
# Strategy: Fetch by month, sleep between requests

BASE_URL = "https://finnhub.io/api/v1/company-news"
API_KEY = os.environ['FINNHUB_API_KEY']

# Generate date ranges (monthly batches)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

date_ranges = []
current = start_date
while current < end_date:
    next_month = current + timedelta(days=30)
    if next_month > end_date:
        next_month = end_date
    date_ranges.append((current, next_month))
    current = next_month

print(f"Collecting {len(date_ranges)} monthly batches...")
print(f"Rate limit: 60 req/min (1 req/sec)")
print()

total_articles = 0
errors = 0

# Sample tickers (S&P 500 major companies)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']

for i, (from_date, to_date) in enumerate(date_ranges[:10], 1):  # Limit to first 10 batches for now
    from_str = from_date.strftime('%Y-%m-%d')
    to_str = to_date.strftime('%Y-%m-%d')
    
    print(f"[{i}/{len(date_ranges)}] Fetching {from_str} to {to_str}...")
    
    batch_articles = []
    
    for ticker in tickers:
        try:
            url = f"{BASE_URL}?symbol={ticker}&from={from_str}&to={to_str}&token={API_KEY}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                articles = response.json()
                
                for article in articles:
                    batch_articles.append({
                        'news_id': article.get('id'),
                        'title': article.get('headline'),
                        'summary': article.get('summary'),
                        'source': article.get('source'),
                        'url': article.get('url'),
                        'published_at': pd.to_datetime(article.get('datetime'), unit='s'),
                        'category': article.get('category'),
                        'related': ticker,
                        'raw_json': json.dumps(article)
                    })
                
                # Rate limiting: 1 req/sec
                time.sleep(1.1)
                
            elif response.status_code == 429:
                print(f"   ⚠️ Rate limit hit, sleeping 60s...")
                time.sleep(60)
                errors += 1
            else:
                print(f"   ⚠️ {ticker}: HTTP {response.status_code}")
                errors += 1
                
        except Exception as e:
            print(f"   ❌ {ticker}: {e}")
            errors += 1
            continue
    
    # Save batch
    if batch_articles:
        df = pd.DataFrame(batch_articles)
        df['fetched_at'] = datetime.now()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['news_id'])
        
        output_dir = BRONZE_DIR / 'finnhub_historical_lakehouse'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'news_{from_str}_to_{to_str}.parquet'
        df.to_parquet(output_file, index=False)
        
        total_articles += len(df)
        print(f"   ✅ Saved {len(df)} articles (Total: {total_articles})")
    else:
        print(f"   ⚠️ No articles in this batch")
    
    print()

print("=" * 80)
print("✅ PHASE 3 COMPLETE!")
print("=" * 80)
print(f"Total articles collected: {total_articles}")
print(f"Errors: {errors}")
print(f"Output: {BRONZE_DIR / 'finnhub_historical_lakehouse'}")
